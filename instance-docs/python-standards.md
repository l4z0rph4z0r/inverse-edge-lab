# Python Development Standards

## Core Principles

1. **Python 3.11+** - Always use modern Python
2. **Type Hints** - Use typing for all functions
3. **Async First** - Default to async for I/O operations
4. **Pydantic** - Use for all data validation
5. **FastAPI** - Preferred web framework (no Flask)

## Code Style

### Imports
```python
# Standard library
import os
import sys
from datetime import datetime
from typing import Optional, List, Dict

# Third party
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Local
from app.models import User
from app.services import UserService
```

### Naming Conventions
```python
# Variables and functions: snake_case
user_name = "John"
def get_user_by_id(user_id: int) -> User:
    pass

# Classes: PascalCase
class UserService:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
```

### Type Hints
```python
# Always use type hints
def process_data(
    input_data: List[Dict[str, Any]],
    validate: bool = True
) -> Optional[ProcessedResult]:
    pass

# Use Union for multiple types
from typing import Union
ResponseType = Union[SuccessResponse, ErrorResponse]
```

## Async Patterns

### Basic Async
```python
import asyncio
import httpx

async def fetch_data(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()
```

### Concurrent Operations
```python
# Good: Concurrent execution
results = await asyncio.gather(
    fetch_user(1),
    fetch_posts(1),
    fetch_comments(1)
)

# Bad: Sequential execution
user = await fetch_user(1)
posts = await fetch_posts(1)
comments = await fetch_comments(1)
```

## Pydantic Models

### Basic Model
```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class User(BaseModel):
    id: int
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    created_at: datetime
    is_active: bool = True
    bio: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
```

### Validation
```python
from pydantic import validator

class Product(BaseModel):
    name: str
    price: float
    quantity: int
    
    @validator('price')
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Price must be positive')
        return v
```

## FastAPI Patterns

### Basic App Structure
```python
from fastapi import FastAPI, HTTPException, Depends
from typing import List

app = FastAPI(
    title="My API",
    version="1.0.0",
    description="API Description"
)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/users/{user_id}", response_model=User)
async def get_user(
    user_id: int,
    db: Database = Depends(get_db)
):
    user = await db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

### Dependency Injection
```python
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Database = Depends(get_db)
) -> User:
    user = await db.get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user

@app.get("/me", response_model=User)
async def read_current_user(
    current_user: User = Depends(get_current_user)
):
    return current_user
```

## Error Handling

### Custom Exceptions
```python
class BusinessError(Exception):
    def __init__(self, message: str, code: str):
        self.message = message
        self.code = code
        super().__init__(self.message)

@app.exception_handler(BusinessError)
async def business_error_handler(request, exc: BusinessError):
    return JSONResponse(
        status_code=400,
        content={"error": exc.code, "message": exc.message}
    )
```

## Testing

### Pytest Structure
```python
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_create_user():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/users",
            json={"username": "test", "email": "test@example.com"}
        )
    assert response.status_code == 201
    assert response.json()["username"] == "test"
```

## Security Best Practices

1. **Never hardcode secrets**
```python
# Bad
API_KEY = "sk-1234567890"

# Good
import os
API_KEY = os.getenv("API_KEY")
```

2. **Validate all inputs**
3. **Use parameterized queries**
4. **Implement rate limiting**
5. **Log security events**

## Performance Tips

1. **Use connection pooling**
2. **Implement caching**
3. **Batch operations when possible**
4. **Profile before optimizing**
5. **Monitor memory usage**

Remember: Clean, typed, async Python is the way!