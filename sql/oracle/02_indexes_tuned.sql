-- Index configuration 'tuned': FK indexes + composite/covering indexes for the workload's common predicates.

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
CREATE INDEX ix_orders_date ON orders (order_date);
CREATE INDEX ix_orders_status_date ON orders (status, order_date);
CREATE INDEX ix_orders_cust_date ON orders (customer_id, order_date, total_amount);
CREATE INDEX ix_items_prod_qty ON order_items (product_id, quantity, unit_price);
CREATE INDEX ix_customers_segment ON customers (segment, nation_id);
CREATE INDEX ix_products_cat_price ON products (category_id, unit_price);
CREATE INDEX ix_payments_status ON payments (status, order_id);
CREATE INDEX ix_reviews_prod_rating ON reviews (product_id, rating);

BEGIN DBMS_STATS.GATHER_SCHEMA_STATS(ownname => USER); END;
/
