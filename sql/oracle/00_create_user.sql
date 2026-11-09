-- Run as SYSDBA (or an admin user) to create the benchmark user.
-- Creates the benchmark user in the FREEPDB1 pluggable database.
-- ALTER SESSION SET CONTAINER = FREEPDB1;   -- when connected to the CDB root

CREATE USER sqlrw IDENTIFIED BY sqlrw
  DEFAULT TABLESPACE users QUOTA UNLIMITED ON users;

GRANT CREATE SESSION, CREATE TABLE, CREATE VIEW, CREATE SEQUENCE TO sqlrw;
-- statistics gathering
GRANT ANALYZE ANY TO sqlrw;
-- EXPLAIN PLAN / DBMS_XPLAN.DISPLAY_CURSOR / V$SQL runtime statistics
GRANT SELECT_CATALOG_ROLE TO sqlrw;
GRANT SELECT ANY DICTIONARY TO sqlrw;
-- optional: cold-cache benchmarking (oracle.flush_buffer_cache: true)
GRANT ALTER SYSTEM TO sqlrw;

EXIT;
