# auth_manager.py
import redis
import json
import secrets
import string
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import hashlib
import os

class AuthManager:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.token_prefix = "token:"
        self.user_prefix = "user:"
        self.admin_prefix = "admin:"
        
    def generate_token(self, length: int = 16) -> str:
        """Generate a secure random token"""
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_admin_if_not_exists(self, username: str = "l4z0rph4z0r", password: str = "HR58k$!B!8D@gQyT"):
        """Create default admin account if it doesn't exist"""
        admin_key = f"{self.admin_prefix}{username}"
        try:
            if self.redis:
                exists = self.redis.exists(admin_key)
                if not exists:
                    admin_data = {
                        "username": username,
                        "password": self.hash_password(password),
                        "role": "admin",
                        "created_at": datetime.utcnow().isoformat(),
                        "is_active": True
                    }
                    self.redis.set(admin_key, json.dumps(admin_data))
                    print(f"Admin user {username} created successfully")
                    return True
                else:
                    print(f"Admin user {username} already exists")
        except Exception as e:
            print(f"Error creating admin: {e}")
            # Fallback - admin always exists for local dev
            return True
        return False
    
    def verify_admin(self, username: str, password: str) -> bool:
        """Verify admin credentials"""
        if not self.redis:
            # Fallback for local development
            return username == "l4z0rph4z0r" and password == "HR58k$!B!8D@gQyT"
            
        try:
            admin_key = f"{self.admin_prefix}{username}"
            admin_data = self.redis.get(admin_key)
            
            if admin_data:
                if isinstance(admin_data, bytes):
                    admin_data = admin_data.decode('utf-8')
                admin = json.loads(admin_data)
                return admin["password"] == self.hash_password(password) and admin.get("is_active", True)
        except Exception as e:
            print(f"Error verifying admin: {e}")
            # Fallback for local development
            return username == "l4z0rph4z0r" and password == "HR58k$!B!8D@gQyT"
        
        return False
    
    def create_access_token(self, 
                          created_by: str,
                          token_name: str,
                          expires_in_days: Optional[int] = None,
                          max_uses: Optional[int] = None) -> Tuple[str, Dict]:
        """Create a new access token"""
        token = self.generate_token()
        
        token_data = {
            "token": token,
            "name": token_name,
            "created_by": created_by,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(days=expires_in_days)).isoformat() if expires_in_days else None,
            "max_uses": max_uses,
            "uses": 0,
            "is_active": True,
            "linked_user": None  # Will be set when token is first used
        }
        
        try:
            if self.redis:
                # Store token data
                token_key = f"{self.token_prefix}{token}"
                self.redis.set(token_key, json.dumps(token_data))
                
                # Also store in admin's token list
                admin_tokens_key = f"admin_tokens:{created_by}"
                self.redis.sadd(admin_tokens_key, token)
                
                print(f"Token created successfully: {token[:8]}...")
                print(f"Token data: {json.dumps(token_data, indent=2)}")
        except Exception as e:
            print(f"Error creating token: {e}")
        
        return token, token_data
    
    def validate_token(self, token: str) -> Optional[Dict]:
        """Validate an access token"""
        if not self.redis:
            print("No Redis connection available")
            return None
        
        try:
            token_key = f"{self.token_prefix}{token}"
            token_data = self.redis.get(token_key)
            
            if not token_data:
                print(f"Token not found: {token[:8]}...")
                return None
            
            if isinstance(token_data, bytes):
                token_data = token_data.decode('utf-8')
                
            token_info = json.loads(token_data)
            print(f"Token found: {token_info['name']}")
            
            # Check if token is active
            if not token_info.get("is_active", True):
                print("Token is not active")
                return None
            
            # Check expiration
            if token_info.get("expires_at"):
                expires_at = datetime.fromisoformat(token_info["expires_at"])
                now = datetime.utcnow()
                if now > expires_at:
                    print(f"Token expired. Expires at: {expires_at}, Now: {now}")
                    token_info["is_active"] = False
                    self.redis.set(token_key, json.dumps(token_info))
                    return None
            
            # Check max uses
            if token_info.get("max_uses"):
                if token_info["uses"] >= token_info["max_uses"]:
                    print(f"Token exceeded max uses: {token_info['uses']} >= {token_info['max_uses']}")
                    return None
            
            print("Token is valid")
            return token_info
            
        except Exception as e:
            print(f"Error validating token: {e}")
            return None
    
    def use_token(self, token: str, username: str) -> Optional[Dict]:
        """Use a token to create/login a user"""
        token_info = self.validate_token(token)
        if not token_info:
            return None
        
        try:
            # Update token usage
            token_info["uses"] += 1
            token_info["last_used"] = datetime.utcnow().isoformat()
            
            # Create user profile if first use
            if not token_info.get("linked_user"):
                user_key = f"{self.user_prefix}{username}"
                
                # Check if username already exists
                if self.redis and self.redis.exists(user_key):
                    print(f"Username {username} already taken")
                    return None
                
                user_data = {
                    "username": username,
                    "token": token,
                    "role": "user",
                    "created_at": datetime.utcnow().isoformat(),
                    "is_active": True,
                    "profile": {
                        "display_name": username,
                        "email": None,
                        "preferences": {
                            "language": "en",
                            "theme": "light"
                        }
                    }
                }
                
                if self.redis:
                    self.redis.set(user_key, json.dumps(user_data))
                    token_info["linked_user"] = username
                    print(f"Created user profile for {username}")
            
            # Save updated token info
            if self.redis:
                token_key = f"{self.token_prefix}{token}"
                self.redis.set(token_key, json.dumps(token_info))
                print(f"Updated token usage: {token_info['uses']} uses")
            
            return token_info
            
        except Exception as e:
            print(f"Error using token: {e}")
            return None
    
    def get_user_profile(self, username: str) -> Optional[Dict]:
        """Get user profile"""
        if not self.redis:
            return None
        
        try:
            user_key = f"{self.user_prefix}{username}"
            user_data = self.redis.get(user_key)
            if user_data:
                if isinstance(user_data, bytes):
                    user_data = user_data.decode('utf-8')
                return json.loads(user_data)
        except Exception as e:
            print(f"Error getting user profile: {e}")
            
        return None
    
    def update_user_profile(self, username: str, updates: Dict) -> bool:
        """Update user profile"""
        user = self.get_user_profile(username)
        if not user:
            return False
        
        try:
            # Update profile fields
            if "profile" in updates:
                user["profile"].update(updates["profile"])
            
            if self.redis:
                user_key = f"{self.user_prefix}{username}"
                self.redis.set(user_key, json.dumps(user))
            return True
        except Exception as e:
            print(f"Error updating user profile: {e}")
            return False
    
    def list_tokens(self, admin_username: str) -> List[Dict]:
        """List all tokens created by an admin"""
        if not self.redis:
            return []
        
        tokens = []
        try:
            admin_tokens_key = f"admin_tokens:{admin_username}"
            token_set = self.redis.smembers(admin_tokens_key)
            
            print(f"Found {len(token_set)} tokens for admin {admin_username}")
            
            for token_item in token_set:
                if isinstance(token_item, bytes):
                    token = token_item.decode('utf-8')
                else:
                    token = token_item
                    
                token_key = f"{self.token_prefix}{token}"
                token_data = self.redis.get(token_key)
                
                if token_data:
                    if isinstance(token_data, bytes):
                        token_data = token_data.decode('utf-8')
                    token_info = json.loads(token_data)
                    tokens.append(token_info)
                    print(f"Found token: {token_info['name']}")
            
        except Exception as e:
            print(f"Error listing tokens: {e}")
        
        return sorted(tokens, key=lambda x: x["created_at"], reverse=True)
    
    def revoke_token(self, token: str, admin_username: str) -> bool:
        """Revoke a token"""
        if not self.redis:
            return False
        
        try:
            token_key = f"{self.token_prefix}{token}"
            token_data = self.redis.get(token_key)
            
            if token_data:
                if isinstance(token_data, bytes):
                    token_data = token_data.decode('utf-8')
                    
                token_info = json.loads(token_data)
                
                # Verify admin owns this token
                if token_info["created_by"] == admin_username:
                    token_info["is_active"] = False
                    token_info["revoked_at"] = datetime.utcnow().isoformat()
                    token_info["revoked_by"] = admin_username
                    self.redis.set(token_key, json.dumps(token_info))
                    print(f"Token {token[:8]}... revoked successfully")
                    return True
                    
        except Exception as e:
            print(f"Error revoking token: {e}")
        
        return False
    
    def get_user_stats(self) -> Dict:
        """Get user statistics"""
        if not self.redis:
            return {"total_users": 0, "active_tokens": 0, "total_tokens": 0}
        
        try:
            # Count users
            user_keys = list(self.redis.scan_iter(f"{self.user_prefix}*"))
            total_users = len(user_keys)
            
            # Count tokens
            token_keys = list(self.redis.scan_iter(f"{self.token_prefix}*"))
            total_tokens = len(token_keys)
            active_tokens = 0
            
            for key in token_keys:
                token_data = self.redis.get(key)
                if token_data:
                    if isinstance(token_data, bytes):
                        token_data = token_data.decode('utf-8')
                    token_info = json.loads(token_data)
                    if token_info.get("is_active", True):
                        active_tokens += 1
            
            stats = {
                "total_users": total_users,
                "active_tokens": active_tokens,
                "total_tokens": total_tokens
            }
            print(f"User stats: {stats}")
            return stats
            
        except Exception as e:
            print(f"Error getting user stats: {e}")
            return {"total_users": 0, "active_tokens": 0, "total_tokens": 0}