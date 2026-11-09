-- Index configuration 'fk_indexes': Primary keys + one index per foreign-key column.

CREATE INDEX ix_nations_region ON nations (region_id);
CREATE INDEX ix_customers_nation ON customers (nation_id);
CREATE INDEX ix_suppliers_nation ON suppliers (nation_id);
CREATE INDEX ix_categories_parent ON categories (parent_category_id);
CREATE INDEX ix_products_category ON products (category_id);
CREATE INDEX ix_products_supplier ON products (supplier_id);
CREATE INDEX ix_orders_customer ON orders (customer_id);
CREATE INDEX ix_items_product ON order_items (product_id);
CREATE INDEX ix_payments_order ON payments (order_id);
CREATE INDEX ix_reviews_product ON reviews (product_id);
CREATE INDEX ix_reviews_customer ON reviews (customer_id);

BEGIN DBMS_STATS.GATHER_SCHEMA_STATS(ownname => USER); END;
/
