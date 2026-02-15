-- PostgreSQL initialization script for CQRS Write Database
-- This script sets up the database schema and replication for CDC

-- Create orders table
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    customer_name VARCHAR(100) NOT NULL,
    customer_email VARCHAR(150) NOT NULL,
    product_name VARCHAR(200) NOT NULL,
    quantity INTEGER NOT NULL DEFAULT 1,
    unit_price NUMERIC(12, 2) NOT NULL,
    total_price NUMERIC(14, 2) NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'pending',
    shipping_address TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_customer_email ON orders(customer_email);
CREATE INDEX IF NOT EXISTS idx_orders_updated_at ON orders(updated_at);

-- Create function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to auto-update updated_at on row update
DROP TRIGGER IF EXISTS update_orders_updated_at ON orders;
CREATE TRIGGER update_orders_updated_at
    BEFORE UPDATE ON orders
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create publication for logical replication (for CDC)
DROP PUBLICATION IF EXISTS cdc_publication;
CREATE PUBLICATION cdc_publication FOR TABLE orders;

-- Grant replication permissions
ALTER TABLE orders REPLICA IDENTITY FULL;

-- Insert some initial seed data
INSERT INTO orders (customer_name, customer_email, product_name, quantity, unit_price, total_price, status, shipping_address, metadata) VALUES
    ('Alice Johnson', 'alice@example.com', 'Mechanical Keyboard', 2, 150000.00, 300000.00, 'completed', 'Jl. Sudirman No. 1, Jakarta', '{"color": "black", "switch": "cherry-mx-blue"}'),
    ('Bob Smith', 'bob@example.com', 'USB-C Hub', 1, 250000.00, 250000.00, 'shipped', 'Jl. Thamrin No. 5, Jakarta', '{"ports": 7, "brand": "Anker"}'),
    ('Charlie Lee', 'charlie@example.com', 'Monitor Stand', 3, 175000.00, 525000.00, 'pending', 'Jl. Ganesha No. 10, Bandung', '{"material": "aluminum", "adjustable": true}');

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'PostgreSQL initialization completed successfully';
    RAISE NOTICE 'Table orders created with % rows', (SELECT COUNT(*) FROM orders);
    RAISE NOTICE 'Publication cdc_publication created for CDC';
END $$;
