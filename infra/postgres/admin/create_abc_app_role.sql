-- Safe to run while apps are connected: creates NOLOGIN owner role only.
-- (Full block also lives in infra/postgres/init/02-abc-app-role.sh.)
--
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'abc_app') THEN
        CREATE ROLE abc_app NOLOGIN;
    END IF;
END
$$;

GRANT CONNECT ON DATABASE abc_shared TO abc_app;
GRANT USAGE, CREATE ON SCHEMA public TO abc_app;

GRANT abc_app TO research_user;
GRANT abc_app TO trader_user;

ALTER DEFAULT PRIVILEGES FOR ROLE abc_app IN SCHEMA public
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO research_user;
ALTER DEFAULT PRIVILEGES FOR ROLE abc_app IN SCHEMA public
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO trader_user;
ALTER DEFAULT PRIVILEGES FOR ROLE abc_app IN SCHEMA public
GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO research_user;
ALTER DEFAULT PRIVILEGES FOR ROLE abc_app IN SCHEMA public
GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO trader_user;
