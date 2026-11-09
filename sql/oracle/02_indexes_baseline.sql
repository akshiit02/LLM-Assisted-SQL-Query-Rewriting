-- Index configuration 'baseline': Primary keys only (no secondary indexes).


BEGIN DBMS_STATS.GATHER_SCHEMA_STATS(ownname => USER); END;
/
