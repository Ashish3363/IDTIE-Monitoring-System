# util.py
import bcrypt

# Password to hash
plain_password = "password123"

# Truncate to 72 bytes (bcrypt limit)
MAX_BCRYPT_LEN = 72
password_bytes = plain_password[:MAX_BCRYPT_LEN].encode('utf-8')

# Generate salt and hash
salt = bcrypt.gensalt(rounds=12)  # you can adjust rounds if needed
hashed_password = bcrypt.hashpw(password_bytes, salt)

# Print results
print(f"Plain password: {plain_password}")
print(f"Hashed password (bcrypt): {hashed_password.decode()}")
