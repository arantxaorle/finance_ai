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
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from pathlib import Path
from io import StringIO
import json
import logging

try:
    from langchain_community.callbacks import get_openai_callback
except ImportError:
    try:
        from langchain.callbacks import get_openai_callback
    except ImportError:
        get_openai_callback = None

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
                    monthly_limit REAL NOT NULL
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

def build_user_snapshot(conn):
    """Create a compact text snapshot of the user's current finances
    to feed into the AI advisor.
    """
    incomes = get_incomes(conn)
    expenses = get_expenses(conn)
    ti, te, sav, rate = compute_summary(incomes, expenses)

    snapshot_lines = []
    snapshot_lines.append(f"Total income recorded: {ti:.2f}")
    snapshot_lines.append(f"Total expenses recorded: {te:.2f}")
    snapshot_lines.append(f"Total savings (income - expenses): {sav:.2f}")
    snapshot_lines.append(f"Overall savings rate: {rate:.2f}%")

    
    if not expenses.empty:
        last_30 = expenses[expenses["date"] >= pd.Timestamp.today() - pd.Timedelta(days=30)]
        if not last_30.empty:
            by_cat = (
                last_30.groupby("category")["amount"]
                .sum()
                .sort_values(ascending=False)
            )
            snapshot_lines.append("")
            snapshot_lines.append("Spending in the last 30 days by category:")
            for cat, amt in by_cat.items():
                snapshot_lines.append(f"- {cat}: {amt:.2f}")

    try:
        budgets_df = pd.read_sql_query("SELECT * FROM budgets", conn)
    except Exception:
        budgets_df = pd.DataFrame()

    if not budgets_df.empty and not expenses.empty:
        start_of_month = pd.Timestamp(date.today().replace(day=1))
        month_exp = expenses[expenses["date"] >= start_of_month]
        by_cat_month = month_exp.groupby("category")["amount"].sum()

        snapshot_lines.append("")
        snapshot_lines.append("Budget vs spending this month:")
        for _, row in budgets_df.iterrows():
            budget_cat = row["category"]
            limit = float(row["monthly_limit"])
            spent = float(by_cat_month.get(budget_cat, 0.0))
            snapshot_lines.append(
                f"- {budget_cat}: budget {limit:.2f}, spent {spent:.2f}"
            )

    return "\n".join(snapshot_lines)


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
def load_retriever(kb_path, openai_api_key):
    if not kb_path.exists():
        with open(kb_path, "w", encoding="utf-8") as f:
            f.write(
                "Investing basics:\n"
                "- Diversify assets\n"
                "- Long-term investing beats timing\n"
                "- Dollar-cost averaging.\n"
            )
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, tiktoken_enabled=False,)
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    text = Path(kb_path).read_text()
    chunks = splitter.split_text(text)
    vs = FAISS.from_texts(chunks, embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": 2})
    return retriever

retriever = load_retriever(KB_PATH, OPENAI_API_KEY) if OPENAI_API_KEY else None

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

st.sidebar.markdown("---")
currency = st.sidebar.selectbox(
    "Currency",
    options=["$", "€", "£", "AED"],
    index=0
)

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

    goals_df = pd.read_sql_query("SELECT * FROM goals", conn)

    st.subheader("Goals snapshot")

    if goals_df.empty:
        st.caption("No goals yet. Add some in the Goals tab.")
    else:

        goals_df["due_date"] = pd.to_datetime(goals_df["due_date"]).dt.date
        goals_df["days_left"] = (goals_df["due_date"] - today).apply(lambda d: d.days)
        goals_df["months_left"] = goals_df["days_left"].apply(
            lambda d: max(int(d / 30), 1)
        )
        goals_df["required_per_month"] = (
            goals_df["target_amount"] / goals_df["months_left"]
        )

        total_required = goals_df["required_per_month"].sum()
        soonest = goals_df.sort_values("due_date").iloc[0]

        g1, g2, g3 = st.columns(3)
        g1.metric("Active goals", len(goals_df))
        g2.metric("Needed / month for goals", f"${total_required:,.2f}")
        g3.metric(
            "Next goal",
            soonest["name"],
            f"Due {soonest['due_date']}",
        )

        #Delete
        st.dataframe(
            goals_df[
                [
                    "name",
                    "target_amount",
                    "due_date",
                    "months_left",
                    "required_per_month",
                ]
            ]
        )

    st.subheader("Spending over time")

    if expenses.empty:
        st.info("No expenses yet.")
    else:
        
        df = expenses.copy()
        df["year"] = df["date"].dt.year
        df["month_period"] = df["date"].dt.to_period("M")  # e.g. 2025-01

        view = st.radio(
            "View by",
            ["All time", "By month", "By year"],
            horizontal=True,
        )


        data = pd.DataFrame(columns=["category", "amount"])
        title = ""

        if view == "All time":
            data = df.groupby("category")["amount"].sum().reset_index()
            title = "All expenses"

        elif view == "By month":
            month_options = sorted(df["month_period"].unique())
            col_month, _ = st.columns([1, 3])
            with col_month:
                selected_month = st.selectbox(
                    "Month",
                    options=month_options,
                    index=len(month_options) - 1,  # latest month
                )

            data = (
                df[df["month_period"] == selected_month]
                .groupby("category")["amount"]
                .sum()
                .reset_index()
            )
            title = f"Month: {selected_month}"

        else:  # "By year"
            year_options = sorted(df["year"].unique())
            col_year, _ = st.columns([1, 3])
            with col_year:
                selected_year = st.selectbox(
                    "Year",
                    options=year_options,
                    index=len(year_options) - 1,  # latest year
                )

            data = (
                df[df["year"] == selected_year]
                .groupby("category")["amount"]
                .sum()
                .reset_index()
            )
            title = f"Year: {selected_year}"


        if data.empty:
            st.info("No expenses for this period.")
        else:
            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    x="category:N",
                    y="amount:Q",
                    color="category:N",
                    tooltip=["category", "amount"],
                )
                .properties(title=title)
            )
            st.altair_chart(chart, use_container_width=True)




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
        
        df["due_date"] = pd.to_datetime(df["due_date"]).dt.date

        
        df["days_left"] = (df["due_date"] - today).apply(lambda d: d.days)

        
        df["months_left"] = df["days_left"].apply(
            lambda d: max(int(d / 30), 1) if d is not None else 1
        )

        
        df["required_per_month"] = df["target_amount"] / df["months_left"]

        st.subheader("Current goals")
        st.dataframe(
            df[
                [
                    "name",
                    "target_amount",
                    "due_date",
                    "months_left",
                    "required_per_month",
                    "notes",
                ]
            ]
        )

        st.subheader("Goal breakdown")
        for _, row in df.iterrows():
            st.markdown(
                f"**{row['name']}**  \n"
                f"- Target: {row['target_amount']:.2f}  \n"
                f"- Due: {row['due_date']} "
                f"({row['months_left']} month(s) left)  \n"
                f"- Needed: ~{row['required_per_month']:.2f} per month"
            )

    st.markdown("---")

    
    with st.form("add_goal"):
        name = st.text_input("Goal name")
        target = st.number_input("Target amount", min_value=0.0, step=100.0)
        due = st.date_input("Due date", today + timedelta(days=180))
        notes = st.text_area("Notes")

        if st.form_submit_button("Save Goal"):
            if not name.strip():
                st.error("Goal name can’t be empty.")
            elif target <= 0:
                st.error("Target amount must be greater than zero.")
            else:
                safe_execute(
                    conn,
                    """
                    INSERT OR REPLACE INTO goals (name, target_amount, due_date, notes)
                    VALUES (?, ?, ?, ?)
                    """,
                    (name.strip(), float(target), due.isoformat(), notes),
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
                st.metric(sym, f"{currency}{price:,.2f}")

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
    st.title("🤖 AI Advisor Chatboot")

    if not OPENAI_API_KEY:
        st.error("Missing OpenAI API key. Set it in your .env or Streamlit secrets.")
    elif retriever is None:
        st.error("RAG retriever not initialized.")
    else:
        user_snapshot = build_user_snapshot(conn)
        persona = st.selectbox(
            "Response style",
            options=["Friendly coach", "Formal analyst", "Concise summary"],
            index=0,
            help="Choose how the AI should talk to you."
        )

        
        col_t, col_p = st.columns(2)
        with col_t:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                help="Higher = more creative, lower = more deterministic."
            )
        with col_p:
            top_p = st.slider(
                "Top-p",
                min_value=0.1,
                max_value=1.0,
                value=1.0,
                step=0.05,
                help="Nucleus sampling; leave at 1.0 if unsure."
            )

        with st.expander("❓ How to use the AI advisor"):
            st.markdown(
            """
            - You can ask questions like *"What is diversification? or ¿How can I improve my budgeting?"*  
            - The AI will use your financial snapshot to give tailored suggestions
            - Paste details of your own situation and ask for a simple explanation.  
            - Remember: this is **not** professional financial advice.
            """
            )

        query = st.text_input("Ask about your money (spending, saving, investing, etc.):")
        use_cache = st.checkbox("Use cached answer if available", value=True)
        force_refresh = st.checkbox("Force fresh answer", value=False)

        if st.button("Ask"):
            if not query.strip():
                st.warning("Please enter a question.")
            else:
                cached = None
                cache_key = query.strip()

                if use_cache and not force_refresh:
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT answer, meta, created_at FROM ai_cache WHERE question = ?",
                        (cache_key,),
                    )
                    row = cur.fetchone()
                    if row:
                        cached = row

                if cached and not force_refresh:
                    answer, meta_json, created_at = cached
                    st.info(f"Using cached answer from {created_at}.")
                    st.markdown("### Answer")
                    st.write(answer)

                    if meta_json:
                        meta = json.loads(meta_json)
                        st.caption(
                            f"Model: {meta.get('model', 'N/A')} • "
                            f"Tokens: {meta.get('total_tokens', 'N/A')} • "
                            f"Cost: ${meta.get('total_cost', 0):.6f}"
                        )
                else:

                    if persona == "Friendly coach":
                        style_instruction = (
                            "Explain in a warm, friendly tone. Use simple language and examples."
                        )
                    elif persona == "Formal analyst":
                        style_instruction = (
                            "Respond like a professional financial analyst, with a clear, structured tone."
                        )
                    else:  
                        style_instruction = (
                            "Be very concise. Focus on key points and bullet lists."
                        )

                    system_prompt = (
                        "You are a holistic personal finance coach. "
                        "You explain concepts clearly for beginners and intermediate investors. "
                        "You are not a licensed financial advisor and you do not give personalized "
                        "investment, legal, or tax advice. You only provide general educational information.\n\n"
                        "In addition to investments, you help the user improve their everyday money habits: "
                        "spending, budgeting, saving, and planning for goals.\n\n"
                        "Below is a snapshot of the user's current finances, built from their own "
                        "income, expense, budget, and goal data. Use it to give specific, practical "
                        "suggestions about how they can improve their spending and savings. "
                        "Talk about things like which categories look high, where they might cut back, "
                        "and how to align spending with goals. Do NOT reveal or mention that you are "
                        "reading a 'snapshot' or a database; just talk naturally about 'your recent spending', "
                        "'your income', and 'your budget'.\n\n"
                        "USER FINANCIAL SNAPSHOT:\n"
                        f"{user_snapshot}\n\n"
                        f"{style_instruction}"
                    )


                    prompt = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                system_prompt
                                + "\n\nUse the following context to answer the user's question:\n{context}"
                            ),
                            (
                                "human",
                                "Question: {question}"
                            ),
                        ]
                    )

                    llm = ChatOpenAI(
                        model="gpt-4o-mini",
                        openai_api_key=OPENAI_API_KEY,
                        temperature=temperature,
                        top_p=top_p,
                    )

                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        retriever=retriever,
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": prompt},
                    )

                    with st.spinner("Thinking..."):
                        if get_openai_callback is not None:
                            with get_openai_callback() as cb:
                                result = qa_chain({"query": query})
                        else:
                            cb = None
                            result = qa_chain({"query": query})

                    answer = result["result"]
                    sources = result.get("source_documents", [])

                    st.markdown("### Answer")
                    st.write(answer)
                    st.markdown("---")

                    if cb is not None:
                        st.caption(
                            f"Tokens — Prompt: {cb.prompt_tokens}, "
                            f"Completion: {cb.completion_tokens}, "
                            f"Total: {cb.total_tokens}  |  "
                            f"Estimated cost: ${cb.total_cost:.6f}"
                        )
                    else:
                        st.caption("Token usage & cost not available in this LangChain version.")

# ------------- BUDGETS -------------
elif page == "💼 Budgets":
    st.title("💼 Monthly Budgets by Category")

    cur = conn.cursor()

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

       
        merged["utilization_pct"] = (
            merged["spent"] / merged["monthly_limit"]
        ).replace([float("inf")], 0.0).fillna(0.0) * 100

        st.subheader("Budget vs. actual")
        st.dataframe(merged)

        st.subheader("Visual comparison")
        chart_df = merged.set_index("category")[["monthly_limit", "spent"]]
        st.bar_chart(chart_df)
        
        st.subheader("Per-category progress")
        for _, row in merged.iterrows():
            pct = row["utilization_pct"]
            st.write(
                f"**{row['category']}** — "
                f"spent {row['spent']:.2f} / {row['monthly_limit']:.2f} "
                f"({pct:.1f}% used)"
            )
            st.progress(min(pct / 100.0, 1.0))

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
