import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
import numpy as np
st.set_page_config(
    page_title="DeepTrendz",  # Custom title for your page
    page_icon="🛒",           # Optional icon for the page
    layout="wide",            # Layout option (default is 'centered')
    initial_sidebar_state="expanded"  # Sidebar state
)

# Load product data from CSV
csv_file_path = 'products.csv'  # Ensure the CSV file is in the same directory or provide the correct path

@st.cache_data(persist=True, show_spinner=False)
def load_product_data():
    try:
        products = pd.read_csv(csv_file_path)
        products['about_product'] = products['about_product'].fillna('')
        products['discounted_price'] = products['discounted_price'].str.replace('₹', '').str.replace(',', '').astype(float)
        products['rating'] = pd.to_numeric(products['rating'], errors='coerce').fillna(0)
        products['rating_count'] = pd.to_numeric(products['rating_count'], errors='coerce').fillna(0)
        return products
    except FileNotFoundError:
        st.error("The products.csv file was not found.")
        st.stop()

products = load_product_data()

# Initialize session state
if 'cart' not in st.session_state:
    st.session_state['cart'] = []
if 'search_history' not in st.session_state:
    st.session_state['search_history'] = []
if 'buy_history' not in st.session_state:
    st.session_state['buy_history'] = []
if 'browse_history' not in st.session_state:
    st.session_state['browse_history'] = []
if 'ratings' not in st.session_state:
    st.session_state['ratings'] = {}
if 'user_logged_in' not in st.session_state:
    st.session_state['user_logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''
if 'true_recommendations' not in st.session_state:
    st.session_state['true_recommendations'] = []  # To store actual liked products for evaluation
if 'recommendation_evaluation' not in st.session_state:
    st.session_state['recommendation_evaluation'] = {
        'precision': None,
        'recall': None,
        'f1': None,
    }

# User Authentication Page
def show_login_page():
    st.title("Welcome to DeepTrendz")
    st.subheader("Please sign in to continue")
    username_input = st.text_input("Username", "")
    if st.button("Login"):
        if username_input:
            st.session_state['user_logged_in'] = True
            st.session_state['username'] = username_input
            st.success(f"Welcome, {username_input}!")
        else:
            st.warning("Please enter a username.")

# Display Purchase History
def show_purchase_history():
    st.subheader("Your Purchase History")
    if st.session_state['buy_history']:
        for product in st.session_state['buy_history']:
            st.write(f"- {product}")
    else:
        st.write("No purchase history available.")

# Evaluate Recommendations
def evaluate_recommendations(recommendations):
    true_labels = [1 if item in st.session_state['buy_history'] else 0 for item in recommendations]
    predicted_labels = [1] * len(recommendations)  # Assuming all recommended items are true recommendations
    
    precision = precision_score(true_labels, predicted_labels, zero_division=1)
    recall = recall_score(true_labels, predicted_labels, zero_division=1)
    f1 = f1_score(true_labels, predicted_labels, zero_division=1)

    st.session_state['recommendation_evaluation'] = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    st.markdown("### Evaluation Metrics")
    st.write(f"**Precision:** {precision:.2f}")
    st.write(f"**Recall:** {recall:.2f}")
    st.write(f"**F1 Score:** {f1:.2f}")

# Main E-commerce Interface
def show_main_interface():
    st.sidebar.header("User Menu")
    st.sidebar.markdown(f"**Logged in as:** {st.session_state['username']}")
    
    if st.sidebar.button("View Purchase History"):
        show_purchase_history()

    if st.sidebar.button("Logout"):
        st.session_state['user_logged_in'] = False
        st.session_state['username'] = ''
        st.session_state['cart'] = []
        st.session_state['buy_history'] = []
        st.success("Logged out successfully.")
    
    st.sidebar.markdown("---")
    st.sidebar.header("Filter Options")
    categories = list(set(products['category']))
    selected_category = st.sidebar.selectbox("Select a Category", ["All"] + categories)
    min_price, max_price = st.sidebar.slider(
        "Price Range", 
        int(products['discounted_price'].min()), 
        int(products['discounted_price'].max()), 
        (0, int(products['discounted_price'].max())), 
        50
    )

    st.title(f"Welcome back, {st.session_state['username']}!")
    st.subheader("Browse our featured products")

    search_query = st.text_input("Search for products", value="", key='search_input', placeholder="Search...")

    def store_search():
        query = st.session_state['search_input'].strip()
        if query and query not in st.session_state['search_history']:
            st.session_state['search_history'].append(query)

    if st.session_state['search_input'] != "":
        store_search()

    # Show search history only if there is any
    if st.session_state['search_history']:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Search History")
        for query in st.session_state['search_history']:
            st.sidebar.write(f"- {query}")
        st.sidebar.markdown("---")

    # Filter products based on the search query, selected category, and price range.
    filtered_products = [
        p for _, p in products.iterrows()
        if search_query.lower() in p['product_name'].lower() and
           (selected_category == "All" or p['category'] == selected_category) and
           (p['discounted_price'] >= min_price and p['discounted_price'] <= max_price)
    ]

    if filtered_products:
        st.markdown("### All Products")
        cols = st.columns(3)
        for index, product in enumerate(filtered_products):
            with cols[index % 3]:
                st.image(product['img_link'], width=150)
                st.write(f"**{product['product_name']}**")
                st.write(f"Discounted Price: ₹{product['discounted_price']}")
                st.write(f"Category: {product['category']}")
                st.write(f"Rating: {'⭐' * int(product['rating'])} ({product['rating_count']} reviews)")
                quantity = st.number_input("Quantity", min_value=1, value=1, key=f"quantity_{product['product_name']}_{index}")
                
                if st.button(f"Add to Cart", key=f"add_{product['product_name']}_{index}"):
                    for _ in range(quantity):
                        st.session_state['cart'].append(product)
                    st.success(f"{product['product_name']} added to cart.")
    else:
        st.warning("No products found. Try a different search term, category, or price range.")

    st.sidebar.markdown("## Shopping Cart")
    if st.session_state['cart']:
        cart_items = {}
        for item in st.session_state['cart']:
            if item['product_name'] in cart_items:
                cart_items[item['product_name']]['quantity'] += 1
            else:
                cart_items[item['product_name']] = {'price': item['discounted_price'], 'quantity': 1}

        total = sum(details['price'] * details['quantity'] for details in cart_items.values())

        for item, details in cart_items.items():
            st.sidebar.write(f"- {item} (₹{details['price']}) x {details['quantity']}")
        
        st.sidebar.write(f"**Total: ₹{total}**")

        if st.sidebar.button("Buy Now"):
            if cart_items:
                for item, details in cart_items.items():
                    for _ in range(details['quantity']):
                        st.session_state['buy_history'].append(item)
                st.sidebar.success("Order confirmed! Thank you for your purchase.")
                st.session_state['cart'] = []
            else:
                st.sidebar.warning("You haven't selected anything to buy.")
    else:
        st.sidebar.info("Your cart is empty. Add some products to proceed.")

    # Recommendation System
    def get_content_based_recommendations(user_history, num_recommendations=5):
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(products['about_product'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        indices = pd.Series(products.index, index=products['product_name']).drop_duplicates()
        recommended_products = set()

        for title in user_history:
            if title in indices.index:
                idx = indices[title]
                sim_scores = list(enumerate(cosine_sim[idx]))
                sim_scores = [(i, score) for i, score in sim_scores if score is not None and (isinstance(score, float) and not np.isnan(score))]
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1:num_recommendations + 1]
                product_indices = [i[0] for i in sim_scores]
                recommended_products.update(products['product_name'].iloc[product_indices])

        return list(recommended_products)

    def get_collaborative_recommendations(user_history, num_recommendations=5):
        if len(st.session_state['buy_history']) < 2:
            return []

        user_vector = np.zeros(len(products))
        for item in user_history:
            if item in products['product_name'].values:
                idx = products[products['product_name'] == item].index[0]
                user_vector[idx] += 1

        kmeans = KMeans(n_clusters=3)
        kmeans.fit(user_vector.reshape(-1, 1))
        cluster_label = kmeans.predict(user_vector.reshape(-1, 1))

        similar_users = np.where(kmeans.labels_ == cluster_label)[0]
        recommended_products = []

        for user_index in similar_users:
            user_items = st.session_state['buy_history']  # This assumes buy history includes what other users bought.
            recommended_products.extend(user_items)

        return list(set(recommended_products))

    def get_combined_recommendations():
        user_history = st.session_state['buy_history'] + st.session_state['search_history']
        content_recommendations = get_content_based_recommendations(user_history)
        collaborative_recommendations = get_collaborative_recommendations(user_history)

        combined_recommendations = list(set(content_recommendations) | set(collaborative_recommendations))
        return combined_recommendations

    if st.sidebar.button("Show Recommendations"):
        if st.session_state['user_logged_in']:
            recommendations = get_combined_recommendations()
            st.session_state['true_recommendations'] = recommendations  # Store true recommendations in session state
            st.markdown("### Recommended Products")
            if recommendations:
                for product in recommendations:
                    product_details = products[products['product_name'] == product].iloc[0]
                    st.write(f"**{product_details['product_name']}** - ₹{product_details['discounted_price']}")
                    st.image(product_details['img_link'], width=150)
                evaluate_recommendations(recommendations)
            else:
                st.warning("No recommendations found.")
        else:
            st.warning("You need to be logged in to see recommendations.")

# Main application logic
if not st.session_state['user_logged_in']:
    show_login_page()
else:
    show_main_interface()
