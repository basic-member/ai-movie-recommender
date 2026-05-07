import bcrypt

class Hash:
    @staticmethod
    def bcrypt(password: str):
        """
        Hash a plain password using bcrypt.
        """
        pwd_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(pwd_bytes, salt).decode('utf-8')

    @staticmethod
    def verify(plain_password: str, hashed_password: str):
        """
        Verify a plain password against a hashed version.
        """
        password_bytes = plain_password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        try:
            return bcrypt.checkpw(password_bytes, hashed_bytes)
        except Exception:
            return False
