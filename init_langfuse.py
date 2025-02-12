import hashlib
import sys
import uuid
from typing import Optional

import asyncpg
import bcrypt
import fire
from asyncpg import Record
from asyncpg.connection import Connection
from loguru import logger


logger.remove(0)
logger.add(
   sys.stderr,
   format="<green>[{level}]</green> <blue>{time:YYYY-MM-DD HH:mm:ss.SS}</blue> | <white>{message}</white>",
   colorize=True,
   level="DEBUG",
)


class CreateNewEntities:
   """Create a new organization, project, and API key in the Langfuse database associated with the provided user."""

   def __init__(
       self,
       database_url: str,
       organization_id: str,
       organization_name: str,
       project_id: str,
       project_name: str,
       default_user_email: str,
       salt: str,
       public_key: str,
       secret_key: str,
       role: str = "OWNER",
       api_key_note: str = "Provisioned API Key",
   ):
       self.database_url = database_url
       self.organization_id = organization_id
       self.organization_name = organization_name
       self.project_id = project_id
       self.project_name = project_name
       self.role = role
       self.api_key_note = api_key_note
       self.default_user_email = default_user_email
       self.salt = salt
       self.public_key = public_key
       self.secret_key = secret_key

   @staticmethod
   async def generate_fast_hashed_secret_key(secret_key: str, salt: str) -> str:
       return hashlib.sha256((secret_key + salt).encode("utf-8")).hexdigest()

   @staticmethod
   async def generate_hashed_secret_key(secret_key: str) -> str:
       return bcrypt.hashpw(secret_key.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

   @staticmethod
   async def get_default_user(conn: Connection, user_email: str) -> Optional[Record]:
       response = await conn.fetch(
           "SELECT * FROM users WHERE email = $1 LIMIT 1;",
           user_email,
       )
       return response[0] if response else None

   @staticmethod
   async def get_organization(conn: Connection, org_id: str) -> Optional[Record]:
       response = await conn.fetch(
           "SELECT * FROM organizations WHERE id = $1 LIMIT 1;",
           org_id,
       )
       return response[0] if response else None

   @staticmethod
   async def get_project(conn: Connection, proj_id: str) -> Optional[Record]:
       response = await conn.fetch(
           "SELECT * FROM projects WHERE id = $1 LIMIT 1;",
           proj_id,
       )
       return response[0] if response else None

   @staticmethod
   async def create_organization(conn: Connection, org_id: str, org_name: str) -> Record:
       response = await conn.fetch(
           "INSERT INTO organizations (id, name) VALUES ($1, $2) RETURNING *;",
           org_id,
           org_name,
       )
       return response[0]

   @staticmethod
   async def create_project(conn: Connection, org_id: str, proj_id: str, proj_name: str) -> Record:
       response = await conn.fetch(
           "INSERT INTO projects (id, name, org_id) VALUES ($1, $2, $3) RETURNING *;",
           proj_id,
           proj_name,
           org_id,
       )
       return response[0]

   @staticmethod
   async def create_membership(conn: Connection, org_id: str, user_id: str, role: str) -> Record:
       response = await conn.fetch(
           "INSERT INTO organization_memberships (id, org_id, user_id, role) VALUES ($1, $2, $3, $4) RETURNING *;",
           str(uuid.uuid4()),
           org_id,
           user_id,
           role,
       )
       return response[0]

   @staticmethod
   async def create_api_key(
       conn: Connection,
       note: str,
       proj_id: str,
       public_key: str,
       hashed_secret_key: str,
       display_secret_key: str,
       fast_hashed_secret_key: str,
   ) -> Record:
       id = str(uuid.uuid4())
       response = await conn.fetch(
           "INSERT INTO api_keys "
           "(id, note, public_key, hashed_secret_key, display_secret_key, project_id, fast_hashed_secret_key) "
           "VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING *;",
           id,
           note,
           public_key,
           hashed_secret_key,
           display_secret_key,
           proj_id,
           fast_hashed_secret_key,
       )
       return response[0]

   @staticmethod
   def get_display_secret_key(secret_key: str) -> str:
       return secret_key[:6] + "..." + secret_key[-4:]

   async def setup_user_and_project(self) -> None:
       conn: Connection = await asyncpg.connect(self.database_url)
       logger.info("Established connection to the database.")

       try:
           async with conn.transaction():
               user = await self.get_default_user(conn, self.default_user_email)
               logger.info(f"Fetched default user: {user}")

               # Check if organization exists first
               organization = await self.get_organization(conn, self.organization_id)
               if not organization:
                   organization = await self.create_organization(conn, self.organization_id, self.organization_name)
                   logger.info(f"Created new organization: {organization}")
               else:
                   logger.info(f"Using existing organization: {organization}")

               # Check if project exists
               project = await self.get_project(conn, self.project_id)
               if not project:
                   project = await self.create_project(conn, organization["id"], self.project_id, self.project_name)
                   logger.info(f"Created new project: {project}")
               else:
                   logger.info(f"Using existing project: {project}")

               # Create membership only if organization is new
               if not organization:
                   await self.create_membership(conn, organization["id"], user["id"], self.role)
                   logger.info("Created membership.")

               hashed_secret_key = await self.generate_hashed_secret_key(self.secret_key)
               fast_hashed_secret_key = await self.generate_fast_hashed_secret_key(self.secret_key, self.salt)
               display_secret_key = self.get_display_secret_key(self.secret_key)
               logger.info("Generated hashed secret key.")

               await self.create_api_key(
                   conn=conn,
                   note=self.api_key_note,
                   proj_id=self.project_id,
                   public_key=self.public_key,
                   hashed_secret_key=hashed_secret_key,
                   display_secret_key=display_secret_key,
                   fast_hashed_secret_key=fast_hashed_secret_key,
               )
               logger.info("Created API key.")

       except Exception as e:
           logger.exception(f"Failed to create project and user: {e}")
           raise
       finally:
           await conn.close()

   async def main(self) -> None:
       try:
           logger.info("Creating project and user...")
           await self.setup_user_and_project()
       except Exception as e:
           logger.exception(f"Failed to create project and user: {e}")


if __name__ == "__main__":
   fire.Fire(CreateNewEntities)