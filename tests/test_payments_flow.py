from uuid import UUID

def test_full_payment_flow_with_fraud_scoring(client):
    # 1. Create a doctor
    resp = client.post(
        "/doctors",
        json={"full_name": "Dr. Test User", "email": "test@example.com"},
    )
    assert resp.status_code == 200
    doctor = resp.json()
    doctor_id = doctor["id"]
    assert UUID(doctor_id)

    # 2. Create a product
    resp = client.post(
        "/products",
        json={
            "name": "Test Course",
            "price_cents": 15000,
            "category": "course",
        },
    )
    assert resp.status_code == 200
    product = resp.json()
    product_id = product["id"]
    assert UUID(product_id)

    # 3. Create a payment token
    resp = client.post(
        "/tokens",
        json={
            "doctor_id": doctor_id,
            "brand": "VISA",
            "last4": "4242",
            "expires_at": "2028-12-31",
        },
    )
    assert resp.status_code == 200
    token = resp.json()
    token_id = token["id"]
    assert UUID(token_id)

    # 4. Create an order
    resp = client.post(
        "/orders",
        json={
            "doctor_id": doctor_id,
            "currency": "USD",
            "items": [
                {"product_id": product_id, "quantity": 1},
            ],
        },
    )
    assert resp.status_code == 200
    order = resp.json()
    order_id = order["id"]
    assert UUID(order_id)
    assert order["total_amount_cents"] == 15000

    # 5. Pay for the order with an idempotency key
    idem_key = "test-idem-key-123"
    resp = client.post(
        "/payments",
        json={
            "order_id": order_id,
            "payment_token_id": token_id,
            "idempotency_key": idem_key,
        },
    )
    assert resp.status_code == 200
    payment = resp.json()
    assert payment["order_id"] == order_id
    assert payment["payment_token_id"] == token_id
    assert payment["amount_cents"] == 15000
    assert payment["fraud_score"] is not None
    assert payment["fraud_decision"] in ("approve", "review", "block")

    # 6. Call the same request again with same idempotency key, should return same payment
    resp2 = client.post(
        "/payments",
        json={
            "order_id": order_id,
            "payment_token_id": token_id,
            "idempotency_key": idem_key,
        },
    )
    assert resp2.status_code == 200
    payment2 = resp2.json()
    assert payment2["id"] == payment["id"]  # no duplicate payment created

    # 7. Check spend-by-doctor endpoint reflects the captured payment (if not blocked)
    resp = client.get("/reports/spend-by-doctor")
    assert resp.status_code == 200
    rows = resp.json()
    assert isinstance(rows, list)
    # There should be at least one row for our doctor if the payment was captured
    if payment["status"] == "captured":
        assert any(row["doctor_id"] == doctor_id for row in rows)

def test_internal_stats_endpoint(client):
    resp = client.get("/internal/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert "total_doctors" in body
    assert "total_orders" in body
    assert "total_payments" in body
    assert "captured_payments" in body
    assert "blocked_payments_fraud" in body

def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
