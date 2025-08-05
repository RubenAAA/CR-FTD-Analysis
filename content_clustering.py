import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import re

def content_clustering(df_ga4, min_users=50, n_clusters=8):
    """
    content clustering -  find patterns in titles
    """
    
    print("===  CONTENT CLUSTERING ===")
    
    # Get page data
    pages = df_ga4.groupby('event_params_page_title').agg({
        'user_pseudo_id': 'nunique',
        'user_segment': lambda x: (x == 'ftd_converted').sum(),
        'section_category': 'first'  # Keep for comparison at end
    }).reset_index()
    
    pages.columns = ['title', 'users', 'ftd', 'manual_cat']
    pages['ftd_rate'] = pages['ftd'] / pages['users']
    
    # Filter by min users
    pages = pages[pages['users'] >= min_users].reset_index(drop=True)
    print(f"📊 {len(pages)} pages with {min_users}+ users")
    
    # Clean titles
    def clean_title(title):
        if pd.isna(title):
            return ""
        title = str(title).lower()
        title = re.sub(r'[^\w\s]', ' ', title)  # Remove special chars
        title = re.sub(r'\d+', '', title)       # Remove numbers
        title = re.sub(r'\s+', ' ', title).strip()
        return title
    
    pages['clean_title'] = pages['title'].apply(clean_title)
    pages = pages[pages['clean_title'] != ''].reset_index(drop=True)
    
    print(f"📝 {len(pages)} pages after cleaning")
    
    # TF-IDF
    tfidf = TfidfVectorizer(
        max_features=100,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    
    X = tfidf.fit_transform(pages['clean_title'])
    features = tfidf.get_feature_names_out()
    
    print(f"🔤 {X.shape[1]} features extracted")
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    pages['cluster'] = kmeans.fit_predict(X.toarray())
    
    print(f"🎯 {n_clusters} clusters created")
    
    # Analyze clusters
    print("\n=== DISCOVERED PATTERNS ===")
    
    for i in range(n_clusters):
        cluster_pages = pages[pages['cluster'] == i]
        if len(cluster_pages) == 0:
            continue
            
        # Get top terms for this cluster
        cluster_center = kmeans.cluster_centers_[i]
        top_indices = cluster_center.argsort()[-8:][::-1]
        top_terms = [features[idx] for idx in top_indices if cluster_center[idx] > 0]
        
        # Stats
        avg_ftd = cluster_pages['ftd_rate'].mean()
        total_users = cluster_pages['users'].sum()
        
        print(f"\n🏷️ CLUSTER {i+1} ({len(cluster_pages)} pages)")
        print(f"   👥 Users: {total_users:,}")
        print(f"   📈 FTD Rate: {avg_ftd:.2%}")
        print(f"   🔑 Top Terms: {', '.join(top_terms)}")
        
        # Sample titles
        sample_titles = cluster_pages.nlargest(3, 'users')['title'].tolist()
        print("   📄 Sample Pages:")
        for j, title in enumerate(sample_titles, 1):
            print(f"      {j}. {title[:70]}...")
    
    # Compare to manual categories (at the end)
    print("\n=== COMPARISON TO MANUAL CATEGORIES ===")
    
    comparison = pd.crosstab(pages['manual_cat'], pages['cluster'])
    print("Manual Category vs Discovered Cluster:")
    print(comparison)
    
    # Show which clusters are "pure" vs "mixed"
    print("\n=== CLUSTER PURITY ===")
    for i in range(n_clusters):
        cluster_pages = pages[pages['cluster'] == i]
        if len(cluster_pages) == 0:
            continue
            
        manual_dist = cluster_pages['manual_cat'].value_counts()
        dominant = manual_dist.index[0]
        purity = manual_dist.iloc[0] / len(cluster_pages)
        
        if purity > 0.8:
            status = f"🎯 Pure {dominant}"
        elif purity > 0.5:
            status = f"🔶 Mostly {dominant}"
        else:
            status = f"🌈 Mixed ({len(manual_dist)} categories)"
            
        print(f"   Cluster {i+1}: {status} ({purity:.0%})")
    
    # Visualize
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Cluster sizes and performance
    plt.subplot(2, 2, 1)
    cluster_stats = pages.groupby('cluster').agg({
        'users': 'sum',
        'ftd_rate': 'mean'
    })
    plt.scatter(cluster_stats['users'], cluster_stats['ftd_rate'], s=100, alpha=0.7)
    plt.xlabel('Total Users')
    plt.ylabel('Avg FTD Rate')
    plt.title('Cluster Size vs Performance')
    for i, (users, ftd) in enumerate(zip(cluster_stats['users'], cluster_stats['ftd_rate'])):
        plt.annotate(f'C{i+1}', (users, ftd), xytext=(5, 5), textcoords='offset points')
    
    # Subplot 2: Manual vs discovered
    plt.subplot(2, 2, 2)
    sns.heatmap(comparison, annot=True, fmt='d', cmap='Blues')
    plt.title('Manual Category vs Cluster')
    plt.ylabel('Manual Category')
    plt.xlabel('Discovered Cluster')
    
    # Subplot 3: Cluster sizes
    plt.subplot(2, 2, 3)
    cluster_sizes = pages['cluster'].value_counts().sort_index()
    plt.bar(range(len(cluster_sizes)), cluster_sizes.values)
    plt.xlabel('Cluster')
    plt.ylabel('Number of Pages')
    plt.title('Cluster Sizes')
    plt.xticks(range(len(cluster_sizes)), [f'C{i+1}' for i in range(len(cluster_sizes))])
    
    # Subplot 4: FTD rates by cluster
    plt.subplot(2, 2, 4)
    cluster_ftd = pages.groupby('cluster')['ftd_rate'].mean()
    plt.bar(range(len(cluster_ftd)), cluster_ftd.values, color='green', alpha=0.7)
    plt.xlabel('Cluster')
    plt.ylabel('Avg FTD Rate')
    plt.title('FTD Rate by Cluster')
    plt.xticks(range(len(cluster_ftd)), [f'C{i+1}' for i in range(len(cluster_ftd))])
    
    plt.tight_layout()
    plt.show()
    
    return pages, kmeans, tfidf

def run_clustering(df_ga4):
    """
    Run the simple clustering
    """
    pages, model, tfidf = content_clustering(df_ga4, min_users=50, n_clusters=6)
    return pages, model, tfidf

