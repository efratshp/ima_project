
## **Features**

* Doctor onboarding
* Tokenized payment methods (no raw card data stored)
* Product & pricing model
* Order + order items
* Idempotent payment execution
* Fraud scoring (Logistic Regression)
* Audit logs for important events
* Rreporting + internal metrics endpoints
* pytest suite for core flows

---

## **Tech Stack**

* **Python 3.11**
* **FastAPI**
* **PostgreSQL**
* **SQLAlchemy**
* **scikit-learn** (for a local in-process fraud model)
* **Docker / docker-compose**
* **pytest**

---

## **Local Development**

### **With Docker (recommended)**

```bash
docker compose up --build
```

This starts:

* `db`   – PostgreSQL on port 5432
* `web`  – FastAPI app on port 8000

When the service is running:

* Swagger UI → [http://localhost:8000/docs](http://localhost:8000/docs)
* Healthcheck → [http://localhost:8000/health](http://localhost:8000/health)

---

### **Without Docker**

1. Start a PostgreSQL instance (e.g. via Docker):

   ```bash
   docker run --name doctor_payments_db \
       -e POSTGRES_DB=doctor_payments_demo \
       -e POSTGRES_USER=postgres \
       -e POSTGRES_PASSWORD=postgres \
       -p 5432:5432 -d postgres:16-alpine
   ```

2. Install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate    # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set the DB URL:

   ```bash
   export DATABASE_URL="postgresql+psycopg2://postgres:postgres@localhost:5432/doctor_payments_demo"
   ```

4. Run the app:

   ```bash
   uvicorn app.main:app --reload
   ```

---

## **High-Level Architecture**

### **Entities**

* **Doctor** – basic profile
* **Product** – course/event with price
* **PaymentToken** – tokenized representation of a card (token, brand, last4, expiry)
* **Order / OrderItem** – order containing one or more products
* **Payment** – payment attempt with fraud score, decision, PSP metadata, and idempotency key
* **AuditLog** – append-only log for tracing business events

### **Fraud Scoring**

A small Logistic Regression model (trained on synthetic data at startup) is used to score each payment.
Features include:

* payment amount
* first-time vs returning doctor
* token age
* night vs day transaction

Decisions:

* `approve`
* `review`
* `block` (marks the payment as failed with `fraud_suspected`)

### **Endpoints (summary)**

* `POST /doctors`
* `POST /products`
* `POST /tokens`
* `POST /orders`
* `POST /payments`
* `GET /reports/spend-by-doctor`
* `GET /internal/stats`
* `GET /fraud/metrics`
* `GET /health`

Full details available in Swagger UI.

---

## **Testing**

Tests live under `tests/` and use `pytest` with FastAPI’s TestClient.

The suite:

* spins up the API against a temporary SQLite database
* runs a complete payment flow end-to-end
* verifies idempotency behavior
* checks internal stats & health endpoints

Run tests with:

```bash
pytest
```

