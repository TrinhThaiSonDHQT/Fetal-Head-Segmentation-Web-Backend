# Backend Tests

This directory contains test scripts for development and debugging.

**These files are NOT required for production deployment.**

## Test Scripts

### `test_server.py`

Quick test script to verify Flask server is working properly.
Run this to test the backend without the frontend.

**Usage:**

```bash
# Make sure the backend server is running first
python test_server.py
```

### `test_error_handling.py`

Comprehensive test script for backend error handling scenarios.
Tests various error cases to verify graceful error handling.

**Usage:**

```bash
# Make sure the backend server is running first
python test_error_handling.py
```

## Running Tests

1. Start the backend server:

```bash
cd ..
python app.py
```

2. In a new terminal, run tests:

```bash
cd tests
python test_server.py
python test_error_handling.py
```

---

**Note:** For production deployment, these test files can be excluded.
