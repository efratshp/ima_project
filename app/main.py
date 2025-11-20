import enum
import logging
import os
from datetime import datetime, date
from typing import Generator, List, Optional
from uuid import uuid4, UUID

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, conint
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    func,
)
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Logging / config
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("doctor-payments")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/doctor_payments_demo",
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OrderStatus(str, enum.Enum):
    pending = "pending"
    paid = "paid"
    failed = "failed"


class PaymentStatus(str, enum.Enum):
    authorized = "authorized"
    captured = "captured"
    failed = "failed"


class AuditEvent(str, enum.Enum):
    doctor_created = "doctor_created"
    product_created = "product_created"
    order_created = "order_created"
    payment_captured = "payment_captured"
    payment_failed = "payment_failed"


class FraudDecision(str, enum.Enum):
    approve = "approve"
    review = "review"
    block = "block"


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------


class Doctor(Base):
    __tablename__ = "doctors"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    full_name = Column(String, nullable=False)
    email = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    tokens = relationship("PaymentToken", back_populates="doctor")
    orders = relationship("Order", back_populates="doctor")


class Product(Base):
    __tablename__ = "products"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    name = Column(String, nullable=False)
    price_cents = Column(Integer, nullable=False)
    category = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PaymentToken(Base):
    __tablename__ = "payment_tokens"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    doctor_id = Column(String, ForeignKey("doctors.id"), nullable=False)
    token = Column(String, unique=True, nullable=False)
    brand = Column(String, nullable=False)
    last4 = Column(String, nullable=False)
    expires_at = Column(Date, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    doctor = relationship("Doctor", back_populates="tokens")


class Order(Base):
    __tablename__ = "orders"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    doctor_id = Column(String, ForeignKey("doctors.id"), nullable=False)
    total_amount_cents = Column(Integer, nullable=False)
    currency = Column(String, default="USD")
    status = Column(Enum(OrderStatus), default=OrderStatus.pending)
    created_at = Column(DateTime, default=datetime.utcnow)

    doctor = relationship("Doctor", back_populates="orders")
    items = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")
    payments = relationship("Payment", back_populates="order")


class OrderItem(Base):
    __tablename__ = "order_items"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    order_id = Column(String, ForeignKey("orders.id"), nullable=False)
    product_id = Column(String, ForeignKey("products.id"), nullable=False)
    quantity = Column(Integer, nullable=False)
    unit_price_cents = Column(Integer, nullable=False)

    order = relationship("Order", back_populates="items")
    product = relationship("Product")


class Payment(Base):
    __tablename__ = "payments"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    order_id = Column(String, ForeignKey("orders.id"), nullable=False)
    payment_token_id = Column(String, ForeignKey("payment_tokens.id"), nullable=False)
    amount_cents = Column(Integer, nullable=False)
    status = Column(Enum(PaymentStatus), default=PaymentStatus.authorized)
    failure_reason = Column(String, nullable=True)

    provider = Column(String, nullable=False, default="mock-gateway")
    provider_payment_id = Column(String, nullable=True)
    idempotency_key = Column(String, nullable=True, unique=True)

    fraud_score = Column(Float, nullable=True)
    fraud_decision = Column(Enum(FraudDecision), nullable=True)
    is_fraud_label = Column(Boolean, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    order = relationship("Order", back_populates="payments")
    payment_token = relationship("PaymentToken")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    event_type = Column(Enum(AuditEvent), nullable=False)
    entity_type = Column(String, nullable=False)
    entity_id = Column(String, nullable=False)
    metadata = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class DoctorCreate(BaseModel):
    full_name: str
    email: str


class DoctorRead(BaseModel):
    id: UUID
    full_name: str
    email: str
    created_at: datetime

    class Config:
        from_attributes = True


class ProductCreate(BaseModel):
    name: str
    price_cents: conint(gt=0)
    category: Optional[str] = None


class ProductRead(BaseModel):
    id: UUID
    name: str
    price_cents: int
    category: Optional[str]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class TokenCreate(BaseModel):
    doctor_id: UUID
    brand: str
    last4: str = Field(..., min_length=4, max_length=4)
    expires_at: date


class TokenRead(BaseModel):
    id: UUID
    doctor_id: UUID
    token: str
    brand: str
    last4: str
    expires_at: date
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class OrderItemCreate(BaseModel):
    product_id: UUID
    quantity: conint(gt=0)


class OrderCreate(BaseModel):
    doctor_id: UUID
    currency: str = "USD"
    items: List[OrderItemCreate]


class OrderRead(BaseModel):
    id: UUID
    doctor_id: UUID
    total_amount_cents: int
    currency: str
    status: OrderStatus
    created_at: datetime

    class Config:
        from_attributes = True


class PaymentCreate(BaseModel):
    order_id: UUID
    payment_token_id: UUID
    idempotency_key: Optional[str] = None


class PaymentRead(BaseModel):
    id: UUID
    order_id: UUID
    payment_token_id: UUID
    amount_cents: int
    status: PaymentStatus
    failure_reason: Optional[str]
    provider: str
    provider_payment_id: Optional[str]
    idempotency_key: Optional[str]
    fraud_score: Optional[float]
    fraud_decision: Optional[FraudDecision]
    is_fraud_label: Optional[bool]
    created_at: datetime

    class Config:
        from_attributes = True


class FraudMetrics(BaseModel):
    labelled_count: int
    precision: Optional[float]
    recall: Optional[float]


# ---------------------------------------------------------------------------
# Fraud model
# ---------------------------------------------------------------------------


class FraudModel:
    """Simple in-memory fraud model for demo purposes.

    In a real system this would live in its own service and be trained offline.
    """

    def __init__(self) -> None:
        self.model = LogisticRegression()
        self._fit_synthetic()

    def _fit_synthetic(self) -> None:
        rng = np.random.default_rng(42)
        n = 500

        amount = rng.uniform(5, 500, size=n)
        is_first = rng.integers(0, 2, size=n)
        token_age = rng.uniform(0, 365, size=n)
        night = rng.integers(0, 2, size=n)

        base_prob = 0.02 + 0.003 * amount + 0.1 * is_first + 0.15 * night
        base_prob = 1 / (1 + np.exp(-(base_prob - 3)))
        y = rng.binomial(1, np.clip(base_prob, 0.01, 0.99))

        X = np.column_stack(
            [
                amount / 1_000.0,
                is_first,
                token_age / 365.0,
                night,
            ]
        )
        self.model.fit(X, y)
        logger.info("Fraud model trained on synthetic data")

    def score(
        self,
        *,
        amount_cents: int,
        is_first_payment: bool,
        token_age_days: int,
        hour_of_day: int,
    ) -> float:
        amount_scaled = amount_cents / 100_000.0
        night = 1 if hour_of_day < 6 or hour_of_day > 22 else 0
        token_age_scaled = min(max(token_age_days, 0), 365) / 365.0

        X = np.array(
            [
                [
                    amount_scaled,
                    1 if is_first_payment else 0,
                    token_age_scaled,
                    night,
                ]
            ]
        )
        proba = self.model.predict_proba(X)[0, 1]
        return float(proba)

    def decision_from_score(self, score: float) -> FraudDecision:
        if score >= 0.9:
            return FraudDecision.block
        if score >= 0.6:
            return FraudDecision.review
        return FraudDecision.approve


fraud_model = FraudModel()


# ---------------------------------------------------------------------------
# FastAPI app / infra
# ---------------------------------------------------------------------------

app = FastAPI(title="Doctor Payments Service", version="0.2.0")


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def log_event(
    db: Session,
    event_type: AuditEvent,
    entity_type: str,
    entity_id: str,
    metadata: Optional[str] = None,
) -> None:
    entry = AuditLog(
        event_type=event_type,
        entity_type=entity_type,
        entity_id=entity_id,
        metadata=metadata,
    )
    db.add(entry)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready")


# ---------------------------------------------------------------------------
# Routes: doctors / products / tokens
# ---------------------------------------------------------------------------


@app.post("/doctors", response_model=DoctorRead)
def create_doctor(payload: DoctorCreate, db: Session = Depends(get_db)) -> Doctor:
    existing = db.query(Doctor).filter_by(email=payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Doctor with this email already exists")

    doctor = Doctor(full_name=payload.full_name, email=payload.email)
    db.add(doctor)
    log_event(db, AuditEvent.doctor_created, "Doctor", doctor.id, metadata=doctor.email)
    db.commit()
    db.refresh(doctor)
    logger.info("Doctor created: %s (%s)", doctor.id, doctor.email)
    return doctor


@app.post("/products", response_model=ProductRead)
def create_product(payload: ProductCreate, db: Session = Depends(get_db)) -> Product:
    product = Product(
        name=payload.name,
        price_cents=payload.price_cents,
        category=payload.category,
    )
    db.add(product)
    log_event(db, AuditEvent.product_created, "Product", product.id, metadata=product.name)
    db.commit()
    db.refresh(product)
    logger.info("Product created: %s (%s)", product.id, product.name)
    return product


@app.post("/tokens", response_model=TokenRead)
def create_token(payload: TokenCreate, db: Session = Depends(get_db)) -> PaymentToken:
    doctor = db.query(Doctor).get(str(payload.doctor_id))
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")

    token_value = f"tok_{uuid4().hex}"
    token = PaymentToken(
        doctor_id=doctor.id,
        token=token_value,
        brand=payload.brand,
        last4=payload.last4,
        expires_at=payload.expires_at,
    )
    db.add(token)
    db.commit()
    db.refresh(token)
    logger.info("Payment token created: %s for doctor %s", token.id, doctor.id)
    return token


# ---------------------------------------------------------------------------
# Routes: orders / payments
# ---------------------------------------------------------------------------


@app.post("/orders", response_model=OrderRead)
def create_order(payload: OrderCreate, db: Session = Depends(get_db)) -> Order:
    doctor = db.query(Doctor).get(str(payload.doctor_id))
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")

    if not payload.items:
        raise HTTPException(status_code=400, detail="Order must have at least one item")

    product_ids = [str(i.product_id) for i in payload.items]
    products = {
        p.id: p for p in db.query(Product).filter(Product.id.in_(product_ids)).all()
    }

    total = 0
    order = Order(
        doctor_id=doctor.id,
        total_amount_cents=0,
        currency=payload.currency,
        status=OrderStatus.pending,
    )
    db.add(order)
    db.flush()

    for item in payload.items:
        product = products.get(str(item.product_id))
        if not product:
            raise HTTPException(
                status_code=404, detail=f"Product {item.product_id} not found"
            )
        line_total = product.price_cents * item.quantity
        total += line_total

        order_item = OrderItem(
            order_id=order.id,
            product_id=product.id,
            quantity=item.quantity,
            unit_price_cents=product.price_cents,
        )
        db.add(order_item)

    order.total_amount_cents = total
    log_event(db, AuditEvent.order_created, "Order", order.id, metadata=str(total))
    db.commit()
    db.refresh(order)
    logger.info("Order created: %s amount_cents=%s", order.id, order.total_amount_cents)
    return order


@app.post("/payments", response_model=PaymentRead)
def create_payment(payload: PaymentCreate, db: Session = Depends(get_db)) -> Payment:
    order = db.query(Order).get(str(payload.order_id))
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    token = db.query(PaymentToken).get(str(payload.payment_token_id))
    if not token or not token.is_active:
        raise HTTPException(status_code=400, detail="Invalid or inactive payment token")

    if order.status == OrderStatus.paid:
        raise HTTPException(status_code=400, detail="Order already paid")

    if payload.idempotency_key:
        existing = (
            db.query(Payment)
            .filter(Payment.idempotency_key == payload.idempotency_key)
            .first()
        )
        if existing:
            logger.info("Returning existing payment for idempotency_key=%s", payload.idempotency_key)
            return existing

    doctor_payment_count = (
        db.query(func.count(Payment.id))
        .join(Order, Payment.order_id == Order.id)
        .filter(Order.doctor_id == order.doctor_id)
        .scalar()
    )
    is_first_payment = doctor_payment_count == 0
    token_age_days = (datetime.utcnow().date() - token.created_at.date()).days
    hour_of_day = datetime.utcnow().hour

    fraud_score = fraud_model.score(
        amount_cents=order.total_amount_cents,
        is_first_payment=is_first_payment,
        token_age_days=token_age_days,
        hour_of_day=hour_of_day,
    )
    decision = fraud_model.decision_from_score(fraud_score)

    if decision == FraudDecision.block:
        status = PaymentStatus.failed
        failure_reason = "fraud_suspected"
    else:
        status = PaymentStatus.captured
        failure_reason = None

    provider_payment_id = f"gw_{uuid4().hex}" if status == PaymentStatus.captured else None

    payment = Payment(
        order_id=order.id,
        payment_token_id=token.id,
        amount_cents=order.total_amount_cents,
        status=status,
        failure_reason=failure_reason,
        provider="mock-gateway",
        provider_payment_id=provider_payment_id,
        idempotency_key=payload.idempotency_key,
        fraud_score=fraud_score,
        fraud_decision=decision,
    )

    if status == PaymentStatus.captured:
        order.status = OrderStatus.paid
        log_event(
            db,
            AuditEvent.payment_captured,
            "Payment",
            payment.id,
            metadata=str(fraud_score),
        )
        logger.info(
            "Payment captured: order=%s payment=%s fraud_score=%.3f decision=%s",
            order.id,
            payment.id,
            fraud_score,
            decision,
        )
    else:
        order.status = OrderStatus.failed
        log_event(
            db,
            AuditEvent.payment_failed,
            "Payment",
            payment.id,
            metadata=failure_reason,
        )
        logger.warning(
            "Payment blocked for suspected fraud: order=%s fraud_score=%.3f decision=%s",
            order.id,
            fraud_score,
            decision,
        )

    db.add(payment)
    db.commit()
    db.refresh(payment)
    return payment


# ---------------------------------------------------------------------------
# Reporting / health
# ---------------------------------------------------------------------------


@app.get("/reports/spend-by-doctor")
def spend_by_doctor(db: Session = Depends(get_db)) -> List[dict]:
    rows = (
        db.query(
            Doctor.id.label("doctor_id"),
            Doctor.full_name,
            func.sum(Payment.amount_cents).label("total_spend_cents"),
        )
        .join(Order, Order.doctor_id == Doctor.id)
        .join(Payment, Payment.order_id == Order.id)
        .filter(Payment.status == PaymentStatus.captured)
        .group_by(Doctor.id, Doctor.full_name)
        .all()
    )

    return [
        {
            "doctor_id": r.doctor_id,
            "full_name": r.full_name,
            "total_spend_cents": int(r.total_spend_cents or 0),
        }
        for r in rows
    ]


@app.get("/internal/stats")
def internal_stats(db: Session = Depends(get_db)) -> dict:
    total_doctors = db.query(func.count(Doctor.id)).scalar()
    total_orders = db.query(func.count(Order.id)).scalar()
    total_payments = db.query(func.count(Payment.id)).scalar()
    captured_payments = (
        db.query(func.count(Payment.id))
        .filter(Payment.status == PaymentStatus.captured)
        .scalar()
    )
    blocked_payments = (
        db.query(func.count(Payment.id))
        .filter(Payment.status == PaymentStatus.failed, Payment.failure_reason == "fraud_suspected")
        .scalar()
    )

    return {
        "total_doctors": total_doctors,
        "total_orders": total_orders,
        "total_payments": total_payments,
        "captured_payments": captured_payments,
        "blocked_payments_fraud": blocked_payments,
    }


@app.get("/fraud/metrics", response_model=FraudMetrics)
def fraud_metrics(db: Session = Depends(get_db)) -> FraudMetrics:
    labelled = db.query(Payment).filter(Payment.is_fraud_label.isnot(None)).all()
    if not labelled:
        return FraudMetrics(labelled_count=0, precision=None, recall=None)

    tp = fp = fn = 0
    for p in labelled:
        predicted_fraud = p.fraud_decision == FraudDecision.block
        actual_fraud = bool(p.is_fraud_label)

        if predicted_fraud and actual_fraud:
            tp += 1
        elif predicted_fraud and not actual_fraud:
            fp += 1
        elif (not predicted_fraud) and actual_fraud:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None

    return FraudMetrics(
        labelled_count=len(labelled),
        precision=precision,
        recall=recall,
    )


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok"}
