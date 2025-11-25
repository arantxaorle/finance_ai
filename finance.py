"""
Personal Finance & Investment Agent
------------------------------------
A unified Streamlit app that merges:
- Budget tracking, expense management, and goals
- Live market data, portfolio analysis, financial news
- AI-powered investment Q&A using Retrieval-Augmented Generation (RAG)

Target users:
- Individuals who want to track their personal finances and investments
- Beginner to intermediate retail investors who want simple tools + explanations
- People who prefer a single dashboard instead of multiple apps/spreadsheets

Dependencies:
pip install streamlit pandas sqlite3 yfinance faiss-cpu openai langchain beautifulsoup4 reportlab python-dotenv
"""

import os
import sqlite3
from datetime import datetime, date, timedelta
import pandas as pd
import altair as alt
import yfinance as yf
import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from pathlib import Path
from io import StringIO
import json
import logging

logging.basicConfig(level=logging.INFO)

def safe_execute(conn, query, params=()):
    """Thin wrapper around cursor.execute with basic error logging."""
    try:
        cur = conn.cursor()
        cur.execute(query, params)
        conn.commit()
        return cur
    except sqlite3.Error as e:
        logging.error("Database error: %s", e)
        st.error(f"Database error: {e}")
        return None

# ------------------ SETUP ------------------
st.set_page_config(page_title="💼 Finance & Investment Agent", layout="wide")
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


DB_PATH = "finance_agent.db"
KB_PATH = Path("finance_guide.txt")
FAISS_DIR = Path("faiss_index")
FAISS_DIR.mkdir(exist_ok=True)

# ------------------ DATABASE ------------------
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    safe_execute(conn, """CREATE TABLE IF NOT EXISTS incomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    amount REAL NOT NULL,
                    source TEXT,
                    date TEXT NOT NULL
                )""")
    safe_execute(conn, """CREATE TABLE IF NOT EXISTS expenses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    amount REAL NOT NULL,
                    category TEXT NOT NULL,
                    description TEXT,
                    date TEXT NOT NULL
                )""")
    safe_execute(conn, """CREATE TABLE IF NOT EXISTS budgets (
                    category TEXT PRIMARY KEY,
                    limit_amount REAL NOT NULL
                )""")
    safe_execute(conn, """CREATE TABLE IF NOT EXISTS goals (
                    name TEXT PRIMARY KEY,
                    target_amount REAL NOT NULL,
                    due_date TEXT NOT NULL,
                    notes TEXT
                )""")

    return conn

# ------------------ DATA HELPERS ------------------
def get_incomes(conn):
    df = pd.read_sql_query("SELECT * FROM incomes ORDER BY date DESC", conn)
    if not df.empty: df['date'] = pd.to_datetime(df['date'])
    return df

def get_expenses(conn):
    df = pd.read_sql_query("SELECT * FROM expenses ORDER BY date DESC", conn)
    if not df.empty: df['date'] = pd.to_datetime(df['date'])
    return df

def compute_summary(incomes, expenses):
    ti = incomes['amount'].sum() if not incomes.empty else 0
    te = expenses['amount'].sum() if not expenses.empty else 0
    sav = ti - te
    rate = round((sav / ti * 100) if ti > 0 else 0, 2)
    return ti, te, sav, rate

# ------------------ INVESTMENT FUNCTIONS ------------------
def fetch_price(symbol):
    if not symbol or not symbol.strip():
        return None
    try:
        ticker = yf.Ticker(symbol.strip())
        data = ticker.history(period="1d")
        if data.empty:
            return None
        return float(data["Close"].iloc[-1])
    except Exception as e:
        logging.error("Error fetching price for %s: %s", symbol, e)
        return None

def get_historical_return(symbol, period="1y"):
    if not symbol or not symbol.strip():
        return None
    try:
        data = yf.Ticker(symbol.strip()).history(period=period)
        if data.empty or len(data["Close"]) < 2:
            return None
        start, end = float(data["Close"].iloc[0]), float(data["Close"].iloc[-1])
        pct = (end - start) / start * 100
        return {"start": start, "end": end, "pct_return": pct}
    except Exception as e:
        logging.error("Error fetching historical return for %s: %s", symbol, e)
        return None

def fetch_financial_news(symbols):
    news = {}
    headers = {"User-Agent": "finance-agent/1.0"}
    for sym in symbols:
        sym = sym.strip()
        if not sym:
            continue
        try:
            url = f"https://finance.yahoo.com/quote/{sym}/news"
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            items = []
            for a in soup.select("a[href*='/news/']"):
                title = a.get_text(strip=True)
                href = a.get("href")
                if not title or not href:
                    continue
                full_url = href if href.startswith("http") else f"https://finance.yahoo.com{href}"
                items.append((title, full_url))
                if len(items) >= 3:
                    break
            news[sym] = items
        except Exception as e:
            logging.error("Error fetching news for %s: %s", sym, e)
            news[sym] = []
    return news

# ------------------ RAG SETUP ------------------
@st.cache_resource(show_spinner=False)
def load_rag_model(kb_path, openai_api_key):
    if not kb_path.exists():
        with open(kb_path, "w", encoding="utf-8") as f:
            f.write("Investing basics:\n- Diversify assets\n- Long-term investing beats timing\n- Dollar-cost averaging.\n")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    loader = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    text = Path(kb_path).read_text()
    chunks = loader.split_text(text)
    vs = FAISS.from_texts(chunks, embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": 2})
    llm = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key, temperature=0.2)
    return RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True)

rag_chain = load_rag_model(KB_PATH, OPENAI_API_KEY) if OPENAI_API_KEY else None

# ------------------ UI ------------------
st.sidebar.title("📚 Navigation")
page = st.sidebar.radio("Select", [
    "🏠 Dashboard",
    "💰 Add Income/Expense",
    "🎯 Goals",
    "💼 Budgets", 
    "📊 Investments",
    "🤖 AI Advisor",
    "📝 Manage Data", 
])

conn = init_db()
today = date.today()

# ------------- DASHBOARD -------------
if page == "🏠 Dashboard":
    st.title("🏠 Financial Overview")

    incomes, expenses = get_incomes(conn), get_expenses(conn)
    ti, te, sav, rate = compute_summary(incomes, expenses)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Income", f"${ti:,.2f}")
    c2.metric("Total Expenses", f"${te:,.2f}")
    c3.metric("Savings", f"${sav:,.2f}")
    c4.metric("Savings Rate", f"{rate}%")

    st.subheader("Expenses by Category")
    if not expenses.empty:
        cat_sum = expenses.groupby("category")["amount"].sum().reset_index()
        chart = alt.Chart(cat_sum).mark_bar().encode(
            x="category:N", y="amount:Q", color="category:N", tooltip=["category", "amount"]
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No expenses yet.")

# ------------- ADD INCOME/EXPENSE -------------
elif page == "💰 Add Income/Expense":
    st.title("💰 Add Transactions")
    ttype = st.radio("Type", ["Income", "Expense"])
    amount = st.number_input("Amount", min_value=0.0, step=10.0)
    desc = st.text_input("Description / Source")
    cat = st.selectbox("Category", ["Housing","Groceries","Transport","Dining","Utilities","Entertainment","Healthcare","Other"])
    d = st.date_input("Date", today)

    if st.button("Save"):
        if ttype == "Income":
            safe_execute(
                conn,
                "INSERT INTO incomes (amount, source, date) VALUES (?, ?, ?)",
                (amount, desc, d.isoformat()),
            )
        else:
            safe_execute(
                conn,
                "INSERT INTO expenses (amount, category, description, date) VALUES (?, ?, ?, ?)",
                (amount, cat, desc, d.isoformat()),
            )
        st.success("Saved successfully!")

# ------------- GOALS -------------
elif page == "🎯 Goals":
    st.title("🎯 Financial Goals")

    df = pd.read_sql_query("SELECT * FROM goals", conn)

    if df.empty:
        st.info("No goals yet.")
    else:
        st.dataframe(df)

    with st.form("add_goal"):
        name = st.text_input("Goal name")
        target = st.number_input("Target amount", min_value=0.0, step=100.0)
        due = st.date_input("Due date", today + timedelta(days=180))
        notes = st.text_area("Notes")

        if st.form_submit_button("Save Goal"):
            safe_execute(
                conn,
                "INSERT OR REPLACE INTO goals (name, target_amount, due_date, notes) VALUES (?, ?, ?, ?)",
                (name, target, due.isoformat(), notes),
            )
            st.success("Goal saved!")

# ------------- INVESTMENTS -------------

elif page == "📊 Investments":
    st.title("📊 Investment Dashboard")
    symbols = st.text_input("Enter stock symbols (comma separated)", "AAPL,MSFT,GOOGL,SPY").upper().split(",")
    symbols = [s.strip() for s in symbols if s.strip()]

    cols = st.columns(min(5, len(symbols)))
    for i, sym in enumerate(symbols):
        with cols[i % len(cols)]:
            price = fetch_price(sym)
            if price is None:
                st.warning(f"Could not fetch price for {sym}. Check the symbol or try again later.")
            else:
                st.metric(sym, f"${price:,.2f}")

    st.subheader("📈 Historical Returns")
    sym = st.selectbox("Select symbol", symbols)
    period = st.selectbox("Period", ["1mo","3mo","6mo","1y","5y"], index=3)
    if st.button("Calculate Return"):
        res = get_historical_return(sym, period)
        if res:
            st.success(f"{sym} Return ({period}): {res['pct_return']:.2f}% (${res['start']:.2f} → ${res['end']:.2f})")

    st.subheader("📰 Financial News")
    if st.button("Fetch Latest News"):
        news = fetch_financial_news(symbols)
        for s, items in news.items():
            st.write(f"### {s}")
            for title, url in items:
                st.write(f"- [{title}]({url})")

# ------------- AI ADVISOR -------------
elif page == "🤖 AI Advisor":
    st.title("🤖 Investment Knowledge Base Chat")
    if not OPENAI_API_KEY:
        st.error("Missing OpenAI API key. Set it in your .env or Streamlit secrets.")
    else:
        query = st.text_input("Ask your investment question:")
        if st.button("Ask"):
            with st.spinner("Thinking..."):
                result = rag_chain({"query": query})
                st.markdown("### Answer")
                st.write(result["result"])
                st.markdown("---")
                for i, doc in enumerate(result["source_documents"]):
                    st.caption(f"Source {i+1}: {doc.page_content[:200]}...")

# ------------- BUDGETS -------------
elif page == "💼 Budgets":
    st.title("💼 Monthly Budgets by Category")

    cur = conn.cursor()

    # Show existing budgets
    df_budgets = pd.read_sql_query("SELECT * FROM budgets", conn)
    if df_budgets.empty:
        st.info("No budgets defined yet.")
    else:
        st.subheader("Current budgets")
        st.dataframe(df_budgets)

    st.markdown("---")
    st.subheader("Add / update a budget")

    with st.form("budget_form"):
        category = st.text_input("Category (e.g. Rent, Groceries, Investing)")
        monthly_limit = st.number_input(
            "Monthly limit", min_value=0.0, step=50.0, format="%.2f"
        )
        submitted = st.form_submit_button("Save budget")

    if submitted:
        if not category.strip():
            st.error("Category name can’t be empty.")
        elif monthly_limit <= 0:
            st.error("Monthly limit must be greater than zero.")
        else:
            res = safe_execute(
                conn,
                """
                INSERT INTO budgets (category, monthly_limit)
                VALUES (?, ?)
                ON CONFLICT(category) DO UPDATE SET monthly_limit=excluded.monthly_limit
                """,
                (category.strip(), float(monthly_limit)),
            )
            if res is not None:
                st.success(f"Budget saved for category: {category.strip()}")
                st.experimental_rerun()

    st.markdown("---")
    st.subheader("Compare budgets vs expenses (this month)")

    # Get this month's expenses per category
    start_of_month = date.today().replace(day=1)
    df_expenses = pd.read_sql_query(
        """
        SELECT category, SUM(amount) AS spent
        FROM expenses
        WHERE date >= ?
        GROUP BY category
        """,
        conn,
        params=(start_of_month.isoformat(),),
    )

    if df_budgets.empty or df_expenses.empty:
        st.info("Need both budgets and expenses to show comparisons.")
    else:
        merged = pd.merge(
            df_budgets, df_expenses, on="category", how="left"
        ).fillna({"spent": 0.0})
        merged["remaining"] = merged["monthly_limit"] - merged["spent"]

        st.dataframe(merged)

        over = merged[merged["remaining"] < 0]
        if not over.empty:
            st.warning(
                "You are over budget in: " + ", ".join(over["category"].tolist())
            )


# ------------- MANAGE DATA -------------
elif page == "📝 Manage Data":
    st.title("📝 Manage Incomes & Expenses")

    conn = init_db()
    cur = conn.cursor()

    tab1, tab2 = st.tabs(["Incomes", "Expenses"])

    # ---- Incomes ----
    with tab1:
        st.subheader("Incomes")
        df_inc = pd.read_sql_query("SELECT * FROM incomes ORDER BY date DESC", conn)
        if df_inc.empty:
            st.info("No incomes recorded yet.")
        else:
            st.dataframe(df_inc)

            selected_id = st.selectbox(
                "Select income ID to edit/delete",
                options=df_inc["id"].tolist(),
            )

            row = df_inc[df_inc["id"] == selected_id].iloc[0]

            with st.form("edit_income_form"):
                new_date = st.date_input(
                    "Date", value=date.fromisoformat(row["date"])
                )
                new_source = st.text_input("Source", value=row["source"])
                new_amount = st.number_input(
                    "Amount", min_value=0.0, value=float(row["amount"]), step=10.0
                )
                col_a, col_b = st.columns(2)
                save_btn = col_a.form_submit_button("Save changes")
                delete_btn = col_b.form_submit_button("Delete")

            if save_btn:
                res = safe_execute(
                    conn,
                    """
                    UPDATE incomes
                    SET date = ?, source = ?, amount = ?
                    WHERE id = ?
                    """,
                    (
                        new_date.isoformat(),
                        new_source.strip(),
                        float(new_amount),
                        int(selected_id),
                    ),
                )
                if res is not None:
                    st.success("Income updated.")
                    st.experimental_rerun()

            if delete_btn:
                try:
                    cur.execute("DELETE FROM incomes WHERE id = ?", (int(selected_id),))
                    conn.commit()
                    st.warning("Income deleted.")
                    st.experimental_rerun()
                except sqlite3.Error as e:
                    st.error(f"Error deleting income: {e}")

    # ---- Expenses ----
    with tab2:
        st.subheader("Expenses")
        df_exp = pd.read_sql_query("SELECT * FROM expenses ORDER BY date DESC", conn)
        if df_exp.empty:
            st.info("No expenses recorded yet.")
        else:
            st.dataframe(df_exp)

            selected_eid = st.selectbox(
                "Select expense ID to edit/delete",
                options=df_exp["id"].tolist(),
            )
            erow = df_exp[df_exp["id"] == selected_eid].iloc[0]

            with st.form("edit_expense_form"):
                new_date = st.date_input(
                    "Date", value=date.fromisoformat(erow["date"])
                )
                new_category = st.text_input("Category", value=erow["category"])
                new_description = st.text_input(
                    "Description", value=erow.get("description", "")
                )
                new_amount = st.number_input(
                    "Amount", min_value=0.0, value=float(erow["amount"]), step=10.0
                )
                col_a, col_b = st.columns(2)
                save_btn_e = col_a.form_submit_button("Save changes")
                delete_btn_e = col_b.form_submit_button("Delete")

            if save_btn_e:
                res = safe_execute(
                    conn,
                    """
                    UPDATE expenses
                    SET date = ?, category = ?, description = ?, amount = ?
                    WHERE id = ?
                    """,
                    (
                        new_date.isoformat(),
                        new_category.strip(),
                        new_description.strip(),
                        float(new_amount),
                        int(selected_eid),
                    ),
                )
                if res is not None:
                    st.success("Expense updated.")
                    st.experimental_rerun()
            

            if delete_btn_e:
                res = safe_execute(
                    conn,
                    "DELETE FROM expenses WHERE id = ?",
                    (int(selected_eid),),
                )
                if res is not None:
                    st.warning("Expense deleted.")
                    st.experimental_rerun()
