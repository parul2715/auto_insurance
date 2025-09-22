--1. Structure & Schema Validation
-- Row  --9134
SELECT COUNT(*) AS row_count FROM insurance;

-- Schema info
EXEC sp_help 'insurance';

--2. Missing Data Analysis
-- Missing or placeholder values per column
SELECT 
    SUM(CASE WHEN [Customer] IS NULL THEN 1 ELSE 0 END) AS missing_customer,
    SUM(CASE WHEN [Education] IS NULL OR [Education] = 'Unknown' THEN 1 ELSE 0 END) AS missing_education,
    SUM(CASE WHEN [Income] IS NULL OR [Income] = 9999 THEN 1 ELSE 0 END) AS missing_income,
    SUM(CASE WHEN [Response] IS NULL THEN 1 ELSE 0 END) AS missing_response
FROM insurance;

--3. Duplicate Records Check

-- Find duplicate Customer IDs
SELECT Customer, COUNT(*) AS dup_count
FROM insurance
GROUP BY Customer
HAVING COUNT(*) > 1;

--4. Outlier Detection
-- Min & Max values
SELECT 
    MIN(Customer_Lifetime_Value) AS min_clv,
    MAX(Customer_Lifetime_Value) AS max_clv,
    MIN(Income) AS min_income,
    MAX(Income) AS max_income,
    MIN(Monthly_Premium_Auto) AS min_premium,
    MAX([Monthly_Premium_Auto]) AS max_premium
FROM insurance;

select * from insurance

-- Check if 99981 is common or unique
SELECT Income, COUNT(*) AS Count
FROM insurance
WHERE Income >= 90000
GROUP BY Income
ORDER BY Income DESC;

--checking emp status for these numbers
SELECT EmploymentStatus, COUNT(*) AS CountHighIncome
FROM insurance
WHERE Income >= 90000
GROUP BY EmploymentStatus
ORDER BY CountHighIncome DESC;

-- Customers above 99th percentile CLV
SELECT *
FROM (
    SELECT *, PERCENTILE_CONT(0.99) 
	WITHIN GROUP (ORDER BY Customer_Lifetime_Value) 
              OVER() AS clv_99
    FROM insurance
) t
WHERE Customer_Lifetime_Value > clv_99;


--5. Data Consistency & Business Logic Validation

-- Summary of Employment Status vs Income validity
SELECT 
    EmploymentStatus,
    SUM(CASE WHEN Income = 0 THEN 1 ELSE 0 END) AS ZeroIncomeCount,
    SUM(CASE WHEN Income <> 0 THEN 1 ELSE 0 END) AS NonZeroIncomeCount,
    COUNT(*) AS Total
FROM insurance
GROUP BY EmploymentStatus
ORDER BY Total DESC;


-- Unexpected values in categorical columns
SELECT DISTINCT Response FROM insurance;
SELECT DISTINCT Coverage FROM insurance;
SELECT DISTINCT Policy_Type FROM insurance;

-- Negative values check
SELECT *
FROM insurance
WHERE [Income] < 0 OR Customer_Lifetime_Value < 0 OR Monthly_Premium_Auto < 0;



-- Business rule: Months Since Last Claim ≤ Months Since Policy Inception
SELECT *
FROM insurance
WHERE Months_Since_Last_Claim > Months_Since_Policy_Inception;

--6. Cardinality & Unique Value Check
SELECT 
    COUNT(DISTINCT Customer) AS unique_customers,
    COUNT(DISTINCT [Policy]) AS unique_policies,
    COUNT(DISTINCT Policy_Type) AS unique_policy_types,
    COUNT(DISTINCT [State]) AS unique_states,
    COUNT(DISTINCT Sales_Channel) AS unique_sales_channels
FROM insurance;

--7. Distribution and Skewness check

SELECT 
    FLOOR(Income/10000)*10000 AS IncomeRange, COUNT(*) AS Count
FROM insurance
GROUP BY FLOOR(Income/10000)*10000
ORDER BY IncomeRange;

-- Check skewness manually by comparing Mean vs Median
WITH Stats AS (
    SELECT 
        AVG(Income) AS MeanIncome
    FROM insurance
),
MedianCalc AS (
    SELECT DISTINCT 
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY Income) 
            OVER () AS MedianIncome
    FROM insurance
)
SELECT s.MeanIncome, m.MedianIncome
FROM Stats s CROSS JOIN MedianCalc m;

--8. Correlation & Multicollinearity Check

SELECT 
    (AVG(CAST(Income AS FLOAT) * CAST(Monthly_Premium_Auto AS FLOAT)) 
     - AVG(CAST(Income AS FLOAT)) * AVG(CAST(Monthly_Premium_Auto AS FLOAT)))
    / (STDEV(CAST(Income AS FLOAT)) * STDEV(CAST(Monthly_Premium_Auto AS FLOAT))) AS Corr_Income_Premium
FROM insurance;

--9. Target Variable Assessment(Response)

SELECT [Response], COUNT(*) AS count_responses
FROM insurance
GROUP BY [Response];

--10. Domain-Specific Business Rules

-- CLV should be proportional to Premium × Tenure --3780 rows
SELECT *
FROM insurance
WHERE Customer_Lifetime_Value < 0 
   OR Customer_Lifetime_Value > (Monthly_Premium_Auto * Months_Since_Policy_Inception * 2);

-- Check "Months Since Policy Inception" validity
SELECT *
FROM insurance
WHERE Months_Since_Policy_Inception < 0 
   OR Months_Since_Policy_Inception > Customer_Lifetime_Value; 

-- Policy Type vs Coverage mapping
SELECT Policy_Type, Coverage, COUNT(*) AS Count
FROM insurance
GROUP BY Policy_Type, Coverage
ORDER BY COUNT(*) DESC;

-- Vehicle size & class consistency
SELECT DISTINCT Vehicle_Size, Vehicle_Class
FROM insurance
ORDER BY Vehicle_Size, Vehicle_Class;

--11. Feature Redundancy Check

-- Check if Policy is redundant to PolicyType
SELECT Policy, Policy_Type, COUNT(*) AS Count
FROM insurance
GROUP BY Policy, Policy_Type
ORDER BY Policy_Type;

select *from insurance

--12. Time-Based Consistency
-- Future effective dates
SELECT *
FROM insurance
WHERE Effective_To_Date > GETDATE();

--13. Scaling & Normalization Needs
-- Compare ranges of numeric features
SELECT 
    MIN(Income) AS min_income, MAX([Income]) AS max_income,
    MIN(Customer_Lifetime_Value) AS min_clv, MAX(Customer_Lifetime_Value) AS max_clv,
    MIN(Monthly_Premium_Auto) AS min_premium, MAX(Monthly_Premium_Auto) AS max_premium,
    MIN(Total_Claim_Amount) AS min_claim, MAX(Total_Claim_Amount) AS max_claim
FROM insurance;

--14. Leakage Prevention Check
-- Check Response column balance (future leakage risk depends on data understanding)
SELECT Response, COUNT(*)
FROM insurance
GROUP BY Response;

SELECT * FROM insurance


----------------data preparation--------------------


--1. Fix Claim Date / Months Errors(1483 rows affected)
-- Step 1: Calculate Median
WITH Ordered AS (
    SELECT Months_Since_Last_Claim,
           ROW_NUMBER() OVER (ORDER BY Months_Since_Last_Claim) AS rn,
           COUNT(*) OVER() AS total_count
    FROM insurance
),
MedianCalc AS (
    SELECT AVG(1.0 * Months_Since_Last_Claim) AS MedianClaim
    FROM Ordered
    WHERE rn IN ((total_count + 1)/2, (total_count + 2)/2)
)
-- Step 2: Replace inconsistent values with median
UPDATE insurance
SET Months_Since_Last_Claim = (SELECT MedianClaim FROM MedianCalc)
WHERE Months_Since_Last_Claim > Months_Since_Policy_Inception;


---2. Fix CLV Issues(9134 rows updated)

---STEP 1 CALCULATE SEGMENT WISE MEDIAN
;WITH Ordered AS (
    SELECT 
        EmploymentStatus,
        Customer_Lifetime_Value,
        ROW_NUMBER() OVER (PARTITION BY EmploymentStatus ORDER BY Customer_Lifetime_Value) AS rn,
        COUNT(*) OVER (PARTITION BY EmploymentStatus) AS total_count
    FROM insurance
)
SELECT 
    EmploymentStatus,
    AVG(1.0 * Customer_Lifetime_Value) AS median_CLV
INTO #Median_CLV_Segments   -- temp table
FROM Ordered
WHERE rn IN ((total_count + 1)/2, (total_count + 2)/2)
GROUP BY EmploymentStatus;

-- Step 2: Update with segment-wise median
ALTER TABLE insurance
ADD CLV_Corrected FLOAT;

UPDATE i
SET i.CLV_Corrected = 
    CASE 
        WHEN i.Customer_Lifetime_Value < 0 
             OR i.Customer_Lifetime_Value > (i.Monthly_Premium_Auto * i.Months_Since_Policy_Inception * 2) 
        THEN m.median_CLV
        ELSE i.Customer_Lifetime_Value
    END
FROM insurance i
JOIN #Median_CLV_Segments m 
    ON i.EmploymentStatus = m.EmploymentStatus;

--actual replacements
SELECT COUNT(*) AS replaced_rows
FROM insurance
WHERE CLV_Corrected <> Customer_Lifetime_Value

--See a sample of replaced values:

SELECT TOP 20 
    EmploymentStatus,
    Customer_Lifetime_Value AS Original_CLV,
    CLV_Corrected AS Corrected_CLV
FROM insurance
WHERE CLV_Corrected <> Customer_Lifetime_Value
ORDER BY ABS(Customer_Lifetime_Value - CLV_Corrected) DESC;


SELECT * FROM insurance
WHERE Monthly_Premium_Auto<=100

