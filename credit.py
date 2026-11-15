import json
import os
import re
import sqlite3

import pandas as pd
import sqlparse
import streamlit as st
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="Northbridge Credit Risk Query Engine", layout="wide")

DB_PATH = "credit_risk_portfolio.db"
MODEL_NAME = "gpt-4o-mini"
EVALUATOR_MODEL_NAME = "gpt-4o-mini"


def load_credentials():
    try:
        return st.secrets["OPENAI_API_KEY"], st.secrets["OPENAI_API_BASE"]
    except Exception:
        with open("config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg["OPENAI_API_KEY"], cfg["OPENAI_API_BASE"]


@st.cache_resource
def init_llms():
    api_key, api_base = load_credentials()
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_BASE_URL"] = api_base
    return (
        ChatOpenAI(model=MODEL_NAME, temperature=0),
        ChatOpenAI(model=EVALUATOR_MODEL_NAME, temperature=0),
    )


@st.cache_resource
def init_connection():
    return sqlite3.connect(
        f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False
    )


llm, evaluator_llm = init_llms()
conn = init_connection()


database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""

sql_1 = """ 
SELECT s.sector_name,
       ROUND(SUM(l.total_outstanding) / 1e6, 2) AS total_outstanding_mn,
       ROUND(SUM(CASE WHEN l.asset_classification IN ('Substandard', 'Doubtful', 'Loss')
                      THEN l.total_outstanding ELSE 0 END) / 1e6, 2) AS npa_outstanding_mn
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
GROUP BY s.sector_name
ORDER BY total_outstanding_mn DESC
"""

sql_2 = """
SELECT loan_category,
       ROUND(SUM(total_outstanding) / 1e6, 2) AS total_outstanding_mn,
       COUNT(*) AS loan_count
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding_mn DESC
"""

sql_3 = """
SELECT ifrs9_stage,
       COUNT(*) AS loan_count,
       ROUND(SUM(ead_amount) / 1e6, 2) AS total_ead_mn,
       ROUND(SUM(ecl_amount) / 1e6, 2) AS total_ecl_mn
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage
"""

sql_4 = """
SELECT s.sector_name,
       ROUND(AVG(p.provision_coverage_ratio), 2) AS avg_provision_coverage_ratio
FROM provisioning p
JOIN loan_master l ON p.loan_account_number = l.loan_account_number
JOIN sector_master s ON l.sector_code = s.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY s.sector_name
ORDER BY avg_provision_coverage_ratio DESC
"""

sql_5 = """
SELECT
    l.borrower_id,
    MAX(l.borrower_name) AS borrower_name,
    GROUP_CONCAT(DISTINCT s.sector_name) AS sectors,
    COUNT(*) AS loan_count,
    ROUND(SUM(l.total_outstanding) / 1e6, 2) AS total_outstanding_mn,
    GROUP_CONCAT(DISTINCT l.asset_classification) AS asset_classifications,
    ROUND(
        SUM(
            CASE
                WHEN l.asset_classification IN ('Substandard', 'Doubtful', 'Loss')
                THEN l.total_outstanding
                ELSE 0
            END
        ) / 1e6, 2
    ) AS npa_outstanding_mn
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
GROUP BY l.borrower_id
ORDER BY SUM(l.total_outstanding) DESC, l.borrower_id
LIMIT 10
"""

sql_6 = """
SELECT group_name,
       COUNT(*) AS loan_count,
       ROUND(SUM(total_outstanding) / 1e6, 2) AS total_outstanding_mn
FROM loan_master
WHERE group_name IS NOT NULL
GROUP BY group_name
ORDER BY SUM(total_outstanding) DESC
LIMIT 5
"""

sql_7 = """
SELECT l.loan_account_number,
       l.borrower_name,
       s.sector_name,
       ROUND(l.total_outstanding / 1e6, 2) AS total_outstanding_mn,
       l.days_past_due,
       l.asset_classification
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
WHERE l.days_past_due > 0
ORDER BY l.days_past_due DESC
"""

sql_8 = """
SELECT CASE WHEN days_past_due = 0 THEN '0 (Current)'
            WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
            WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
            WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
            ELSE '90+' END AS dpd_bucket,
       COUNT(*) AS loan_count,
       ROUND(SUM(total_outstanding) / 1e6, 2) AS total_outstanding_mn
FROM loan_master
GROUP BY dpd_bucket
ORDER BY MIN(days_past_due)
"""

sql_9 = """
SELECT borrower_id,
       previous_rating,
       internal_rating AS current_rating,
       pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
  AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC
"""

sql_10 = """
SELECT reporting_date,
       ROUND(SUM(ecl_amount) / 1e6, 2) AS total_ecl_mn
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date
"""

verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': sql_1
    },


    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': sql_2
    },


    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': sql_3
    },


    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': sql_4
    },


    'VQ5': {
        'description': 'Top 10 borrowers ranked by aggregated outstanding exposure across all their loans',
        'sql': sql_5
    },


    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': sql_6
    },


    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': sql_7
    },


    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': sql_8
    },


    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': sql_9
    },


    'VQ10': {
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': sql_10
    }
}

def classify_intent(user_question, query_library):

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f""" 
    You are the routing component of a credit-risk query engine at a commercial bank.
Decide whether the user's question can be answered by one of the pre-approved
verified query templates listed below, or whether new SQL must be generated.

### DECISION RULES

1. Choose "verified" when a single template answers the question. A question does not
   need to repeat a template's details word for word. If the question asks for the same
   report in everyday wording (for example "top 10 borrower exposures" for a template that
   lists the largest borrowers by aggregated outstanding amount, or "aging profile" for a days-past-due
   bucket report), choose that template. Do not reject a match just because the question
   leaves out the metric, as long as the template's metric is the standard reading.
2. Templates are fixed and cannot be modified. Each one returns its complete result set,
   and a later step picks out the part the user asked about. So a question that asks
   about one sector, one category, or the highest or lowest item is still "verified" if
   the template returns that metric for all sectors, categories, or items.
3. Choose "generated" only when the question needs something no template provides:
   - a different metric (for example average interest rate, or restructured loans)
   - a different breakdown (for example a different grouping column)
   - an extra filter (for example one specific IFRS 9 stage, one product type, one state)
   - a different time period (templates cover only the periods stated in their description)
4. A template that is only topically similar is NOT a match. Do not pick a template just
   because it shares a keyword with the question. For example, a question about Stage 3
   ECL only is not answered by a template that totals ECL across all stages, and a
   question about average interest rate is not answered by a template about provision
   coverage.
5. When you are genuinely torn between a template and generated SQL because the question
   asks for something the template does not return, choose "generated". Do not use this
   rule when a template clearly matches the question's subject and wording.
6. Use only the query IDs listed below. Never invent a new ID.

### USER QUESTION
{user_question}
### VERIFIED QUERY TEMPLATES
{library_descriptions}

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}

def generate_query(user_question, schema_context):

    generation_prompt = f"""
You are a senior credit-risk data analyst writing SQL for a commercial bank's
read-only analytical database (SQLite). Write ONE SQL query that answers the
user's question, using only the schema provided below.

### RULES

1. Output a single SELECT statement. Never write INSERT, UPDATE, DELETE, DROP, ALTER,
   CREATE, REPLACE, PRAGMA, ATTACH, or any statement that changes data or settings.
2. Use ONLY the tables and columns listed in the schema. Never invent a table or column.
3. Use SQLite syntax and functions only (for example ROUND, CASE WHEN, strftime).
4. Join tables only on the keys shown in the schema:
   - loan_master.sector_code = sector_master.sector_code
   - provisioning.loan_account_number = loan_master.loan_account_number
   - borrower_rating.borrower_id = loan_master.borrower_id
5. When the question mentions a sector, join sector_master and use sector_name.
   Show the sector name, never only the sector code.
6. Business definitions:
   - NPA (non-performing assets) means asset_classification IN ('Substandard', 'Doubtful', 'Loss').
   - A loan is "impaired" when asset_classification IN ('Substandard', 'Doubtful', 'Loss').
   - Stage 3 means ifrs9_stage = 3 in the provisioning table. It is defined by IFRS 9
     stage, not by days past due. The Stage 3 book size is the number of Stage 3 loans
     and their ead_amount, and its expected credit loss is the sum of ecl_amount.
   - "Impaired" is NOT synonymous with IFRS 9 Stage 3 in this project.
     Never use provisioning.ifrs9_stage = 3 to flag a loan as impaired.
     For impairment, use loan_master.asset_classification instead.
   - Restructured loans means is_restructured = 1.
   - Overdue loans means days_past_due > 0.
   - Exposure or outstanding means total_outstanding from loan_master.
7. Time periods:
   - If the question asks about a trend, change over time, "the last four quarters", or
     all quarters, do NOT filter by date. Select reporting_date itself as the period
     column and GROUP BY reporting_date. Do not create a separate quarter label.
   - If the question asks about a point-in-time metric from provisioning or
     borrower_rating and gives no date, use the latest date: '2025-09-30'.
   - If the question names a specific quarter, filter on that date.
8. Dollar amounts: convert to millions with ROUND(x / 1e6, 2) and give the result an alias
   with an _mn suffix. The _mn name is only an alias that you create. No table has a
   column ending in _mn, so never select or read one.
   WRONG: SELECT lm.total_outstanding_mn FROM loan_master lm
   RIGHT: SELECT ROUND(lm.total_outstanding / 1e6, 2) AS total_outstanding_mn FROM loan_master lm
   If the question does not need a dollar amount, do not include one.
   Ratios, rates and percentages: ROUND(x, 2).
9. Averages are simple (unweighted) averages unless the question asks for a weighted one.
10. Give every output column a clear alias. Include counts (COUNT(*) AS loan_count)
    when the question asks how many, and add an ORDER BY that suits the question.
11. If the question asks for a list of items AND an overall figure (a total or an overall
    percentage), return one row per item and compute the overall figure with a window
    function so it is not split per row. An overall percentage of flagged items is the
    number of flagged items divided by ALL items in the list, so the denominator is
    COUNT(*) OVER (). For example:
    ROUND(100.0 * SUM(CASE WHEN flag_condition THEN 1 ELSE 0 END) OVER () / COUNT(*) OVER (), 2) AS overall_pct
    Do not use GROUP BY on the item key just to compute an overall figure.
12. For a question about one IFRS 9 stage over time, use the provisioning table only and
    filter on ifrs9_stage. Do not add any filter the question does not ask for. Pattern:
    SELECT reporting_date, COUNT(*) AS stage3_loan_count,
           ROUND(SUM(ead_amount) / 1e6, 2) AS stage3_ead_mn,
           ROUND(SUM(ecl_amount) / 1e6, 2) AS stage3_ecl_mn
    FROM provisioning WHERE ifrs9_stage = 3
    GROUP BY reporting_date ORDER BY reporting_date
13. Return ONLY the SQL query. No explanation, no comments, no markdown code fences,
    and no trailing semicolon.

### USER QUESTION
{user_question}

### DATABASE SCHEMA
{schema_context}
"""

    sql = llm.invoke(generation_prompt).content.strip()
    
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql

def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):

    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }

    
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result
    if ';' in candidate_sql.rstrip(';').rstrip():
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result

    
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or 'Name' in str(t.ttype)]
    referenced_identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', candidate_sql.lower())
    sql_keywords = {'select','from','where','and','or','group','by','order','having','limit','join','on','as','case',
                    'when','then','else','end','sum','count','avg','min','max','round','desc','asc','left','right',
                    'inner','outer','distinct','null','is','not','in','like','with','union','all','between','coalesce'}
    unknown = [tok for tok in referenced_identifiers
               if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
               and not tok.isdigit() and tok not in ('s', 'l', 'p', 'r', 'e6')]

    
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result

    
    is_verified_track = query_id is not None and query_id in query_library

    if is_verified_track:
        relevance_prompt = f"""
You are checking whether a pre-approved report template can answer a business question
at a commercial bank.

### TEMPLATE DESCRIPTION
{query_library[query_id]['description']}

### USER QUESTION
{user_question}

### RULES

- The template is a standard report that returns a full breakdown (for example all
  sectors, all buckets or all quarters). The user may ask about only one part of it, such
  as a single sector, only the overdue rows, or the highest item. A later step selects
  that part, so a question about one part is still answered by the template.
- Answer verdict "yes" whenever the template's subject and metric are what the question
  needs to be answered.
- Answer verdict "no" only when the template covers a different subject or metric than
  the question (for example the question is about interest rates but the template is
  about provision coverage).
- Do not consider filters, extra rows, row limits or sort order.

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}

"""
    else:
        relevance_prompt = f"""
You are a senior credit-risk analyst reviewing a SQL query before it is run against a
bank's loan database. This SQL was freshly generated for the question below. Decide
whether it correctly answers the question.

### WHAT TO CHECK

1. Metric: does the query calculate the metric the user asked for (for example average
   interest rate versus total outstanding, or ECL versus provision held)?
2. Tables and joins: are the right tables used, and are they joined on the correct keys?
3. Aggregation and grouping: are the right aggregate function (SUM, COUNT, AVG) and the
   right GROUP BY used for the breakdown the user asked for?
4. Filters: are the filters that the question states or clearly implies present? Do not
   require any filter that the question does not mention. For example, do not require an
   overdue or days-past-due filter unless the question asks about overdue loans.
5. Business definitions:
   - NPA and impaired mean asset_classification IN ('Substandard', 'Doubtful', 'Loss').
   - Stage 3 means ifrs9_stage = 3 in the provisioning table. It is defined by IFRS 9
     stage, not by days past due. Stage 3 book size means the Stage 3 loan count and
     ead_amount, and its expected credit loss means the sum of ecl_amount.
   - Restructured means is_restructured = 1.
   - Overdue means days_past_due > 0.
6. Date logic: the provisioning table has exactly four quarter-end reporting dates
   (2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30), and each reporting_date is one
   quarter. For a trend, "over the last four quarters", or "all quarters" question, the
   correct approach is to include all four dates, either by grouping by reporting_date or
   by listing the four dates. That is already a quarterly breakdown. Do not require a
   different date format or a separate quarter label, and do not require the latest date
   to be added separately. For a point-in-time question with no date given, using
   2025-09-30 is correct.
7. Output: does the query return columns that let the user see the answer, such as
   sector names rather than only codes?
8. Units: if a column name says millions (an _mn suffix), the SQL must divide that value
   by 1e6. A column labelled in millions that holds raw dollar amounts is an error.
9. Overall figures: if the question asks for an overall total or percentage alongside a
   list, the overall figure must be computed across all rows (for example with a window
   function). A percentage that is calculated per row, giving only 0 or 100, is an error.
   An overall percentage of flagged items is the number of flagged items divided by ALL
   items in the list. So SUM(CASE WHEN condition THEN 1 ELSE 0 END) OVER () divided by
   COUNT(*) OVER () is the CORRECT calculation. Using COUNT(*) as the denominator is
   correct, and you must not require a conditional count in the denominator.

### HOW TO SCORE

- verdict "yes" when the query would give the user a correct answer to the question.
- verdict "no" only for a concrete error that you can name, such as the wrong metric,
  the wrong table, a missing filter the question requires, or wrong units. Do not fail a
  query for a requirement that the question did not state.
- confidence from 0.0 to 1.0: 0.9 to 1.0 means clearly correct, 0.6 to 0.89 means correct
  with minor doubts, and below 0.6 means probably wrong or too uncertain to trust.
- Judge only whether the SQL answers the question. Do not penalise style, aliases or
  formatting.

### USER QUESTION
{user_question}

### SQL TO REVIEW
{candidate_sql}

### DATABASE SCHEMA
{database_schema}

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}

"""

    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)
        if relevance_json.get('verdict') == 'no' or relevance_json.get('confidence', 0.0) < 0.6:
            result['failed_check'] = 'llm_relevance'
            result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql']
        try:
            expected_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({expected_sql}) LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({candidate_sql}) LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected {len(expected_cols)} columns, got {len(actual_cols)}'
                return result
        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result

def retry_generation(user_question, failed_sql, error_message, schema_context):

    retry_prompt = f"""
You are a senior credit-risk data analyst. A SQL query you wrote for a bank's read-only
SQLite database failed validation. Fix it so that it answers the user's original question.

### HOW TO FIX IT

1. Read the validation error and fix exactly what it describes:
   - "no such column" or "no such table": replace it with a real column or table from the
     schema below, and add the join if the column belongs to another table.
   - A syntax or parsing error: correct the SQL syntax for SQLite.
   - A relevance failure (wrong metric, grouping, filter or date logic): rewrite the query
     so it calculates what the question actually asks for.
2. Keep the original intent. Do not change what the user asked, and do not answer an
   easier question to make the error disappear.
3. Use ONLY the tables and columns in the schema. Join only on these keys:
   - loan_master.sector_code = sector_master.sector_code
   - provisioning.loan_account_number = loan_master.loan_account_number
   - borrower_rating.borrower_id = loan_master.borrower_id
4. Business definitions:
   - NPA means asset_classification IN ('Substandard', 'Doubtful', 'Loss').
   - Impaired means loan_master.asset_classification IN ('Substandard', 'Doubtful', 'Loss').
     IFRS 9 Stage 3 is not an interchangeable impairment flag in this project.
   - Restructured means is_restructured = 1. Overdue means days_past_due > 0.
   - Exposure or outstanding means total_outstanding from loan_master.
5. Time periods: for trends or all-quarters questions do not filter by date. For a
   point-in-time provisioning or rating question with no date given, use '2025-09-30'.
6. Dollar amounts must be shown in millions: write ROUND(SUM(column) / 1e6, 2) and name
   the column with an _mn suffix. Never name a column with _mn unless the value is
   divided by 1e6, and never divide by 1e6 without the _mn suffix. Ratios and rates:
   ROUND(x, 2). Averages are simple, unweighted averages.
7. The query must be a single read-only SELECT statement. No INSERT, UPDATE, DELETE, DROP,
   ALTER, CREATE, REPLACE, ATTACH, or PRAGMA.
8. Return ONLY the corrected SQL. No explanation, no comments, no markdown code fences,
   and no trailing semicolon.

{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql

def execute_query(validated_sql, db_connection):

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result

def generate_response(user_question, dataframe, route, query_id=None):

    facts = [f"Number of rows in the table: {len(dataframe)}"]
    for col in dataframe.columns:
        if pd.api.types.is_numeric_dtype(dataframe[col]) and set(dataframe[col].dropna().unique()) <= {0, 1}:
            facts.append(f"Column {col}: {int(dataframe[col].sum())} of {len(dataframe)} rows have the value 1")
    table_facts = "\n".join(facts)

    response_prompt = f"""
You are a credit-risk analyst at a commercial bank. Write a short, accurate answer to a
business user's question, based only on the query result table below.

### RULES

1. Answer the question directly in the first sentence, with the exact figures.
2. Use only numbers that appear in the table or in the TABLE FACTS. You may do simple
   arithmetic that directly answers the question (a total, a difference, a share of a
   total), but never estimate, guess, or bring in outside facts.
3. Never count rows yourself. When the answer needs the number of items (loans, sectors,
   borrowers) or the number of flagged items, use the TABLE FACTS. When the table has one
   row per item, the number of items is the number of rows in the table.
4. The table can contain more rows than the question asks about. Mention only the rows and
   columns that are relevant:
   - If the question is about one sector, category or stage, report that row.
   - If it asks for the highest, lowest or top items, name those.
   - For a trend, give the starting value, the latest value, and the direction of change.
   - For a distribution, give the largest groups and the total.
5. Units: a column ending in _mn is in millions of dollars (write "$134.33 million").
   Interest rates, ratios and coverage figures are percentages. Counts are numbers of loans.
   Keep the exact figures from the table.
6. If the question depends on a definition, add one short note. NPA means loans classified
   Substandard, Doubtful or Loss. Interest rate averages are simple averages across loans,
   not weighted by exposure.
7. If the table is empty, say that no matching records were found. Do not make up an answer.
8. Write 2 to 4 sentences in plain business language. Do not mention SQL, tables, columns,
   templates, queries, or how the answer was produced. Do not give recommendations unless
   the question asks for them.

### CONTEXT
Route: {route}. If the route is verified, the table is a standard report that may cover
all sectors, categories or periods, so select only what the question asks for.

### USER QUESTION
{user_question}

### TABLE FACTS
{table_facts}

### QUERY RESULT
{dataframe.to_string()}

"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative

def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True):

    log = {
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'narrative': None
    }

    
    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    if verbose:
        print(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
        print(f"    Reason: {log['match_reason']}")

    
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log['candidate_sql'] = candidate_sql

    if verbose:
        print(f"[2] Query Construction: {'loaded from library' if log['route']=='verified' else 'generated fresh SQL'}")

    
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    if verbose:
        print(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            print(f"    Failed check: {gate.get('failed_check')}")
            print(f"    Details: {gate.get('details')}")

    
    if not gate['passed'] and log['route'] == 'generated':
        if verbose:
            print(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        if verbose:
            print(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
            if not gate['passed']:
                print(f"    Retry failed check: {gate.get('failed_check')}")
                print(f"    Retry details: {gate.get('details')}")

    
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        if verbose:
            print(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, **log}

    
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)

    if verbose:
        print(f"[4] Execute: {len(df)} rows returned")
        if exec_result['warnings']:
            print(f"    Warnings: {exec_result['warnings']}")

    
    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative

    
    log['confidence'] = gate.get('relevance_confidence')

    if verbose:
        print(f"[6] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, **log}

EXAMPLE_QUESTIONS = {
    "Test 1: Real estate NPA": "How much of our book is in real estate, and how much of that is non-performing?",
    "Test 2: DPD buckets": "Show me how our overdue book is distributed across DPD buckets.",
    "Test 3: Restructured loans": "List every restructured loan, flag which are impaired, and show the overall impaired percentage.",
    "Test 4: Avg interest rate": "Show me the average interest rate by sector, sorted from highest to lowest.",
    "Test 5: Stage 3 trend": "How has the Stage 3 book size and its expected credit loss moved over the last four quarters?",
}

st.title("Northbridge Bank: Credit Risk Query Engine")
st.caption(
    "Ask a question about the loan portfolio in plain English. "
    "Answers come from pre-approved verified queries or validated generated SQL."
)

if "question" not in st.session_state:
    st.session_state["question"] = ""

st.subheader("Example questions")
cols = st.columns(len(EXAMPLE_QUESTIONS))
for col, (label, q) in zip(cols, EXAMPLE_QUESTIONS.items()):
    if col.button(label, use_container_width=True):
        st.session_state["question"] = q

question = st.text_area("Your question", key="question", height=80)

if st.button("Run query", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Running the pipeline..."):
                out = run_pipeline(
                    question.strip(),
                    conn,
                    verified_query_library,
                    database_schema,
                    verbose=False,
                )

            if out["escalated"]:
                st.error(out["narrative"])
            else:
                st.subheader("Answer")
                st.write(out["narrative"])

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Route", out["route"])
            m2.metric("Template", out["query_id"] or "none")
            m3.metric("Confidence", str(out["confidence"]))
            m4.metric(
                "Rows returned",
                out["row_count"] if out["row_count"] is not None else "n/a",
            )

            if out["dataframe"] is not None:
                st.subheader("Result data")
                st.dataframe(out["dataframe"], use_container_width=True)

            with st.expander("SQL used"):
                st.code(out["executed_sql"] or out["candidate_sql"] or "", language="sql")

            with st.expander("Pipeline log"):
                st.write(f"**Routing reason:** {out['match_reason']}")
                st.write(f"**Retry used:** {out['retry_used']}")
                st.write(f"**Validation outcome:** {out['gate_result']['details']}")
                st.write(f"**Failed check:** {out['gate_result'].get('failed_check')}")
                st.write(f"**Escalated:** {out['escalated']}")
                if out["executed_sql"]:
                    st.write("**Executed SQL:**")
                    st.code(out["executed_sql"], language="sql")
                st.json({
                    "route": out["route"],
                    "query_id": out["query_id"],
                    "match_reason": out["match_reason"],
                    "gate_result": out["gate_result"],
                    "retry_used": out["retry_used"],
                    "escalated": out["escalated"],
                    "row_count": out["row_count"],
                    "confidence": out["confidence"],
                })
        except Exception as exc:
            st.error(f"The query could not be completed: {exc}")
