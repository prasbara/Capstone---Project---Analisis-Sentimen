import pandas as pd
import json
import re
import networkx as nx
from datetime import timedelta
# Removed external community library to avoid dependency error
from collections import Counter

INPUT_FILE = "output_comments.csv"
OUTPUT_JSON = "sna_graph.json"

def clean_username(u):
    return str(u).replace('@', '').strip().lower()

def main():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # 1. Preprocessing
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['username'] = df['username'].apply(clean_username)
    
    # Create Graph
    G = nx.Graph() # Undirected for co-occurrence
    
    # 2. Add Nodes
    print("Building Nodes...")
    user_counts = df['username'].value_counts()
    user_sentiments = df.groupby('username')['sentiment'].agg(lambda x: x.mode().iloc[0])
    
    for user, count in user_counts.items():
        sent = user_sentiments[user]
        # Assign color based on sentiment
        color = "#808080" # Netral
        if sent == 'Positif': color = "#4CAF50"
        if sent == 'Negatif': color = "#F44336"
        
        G.add_node(user, size=int(count), sentiment=sent, color=color, label=user)

    # 3. Add Edges (Mentions)
    print("Building Mention Edges...")
    mention_count = 0
    for idx, row in df.iterrows():
        user = row['username']
        text = str(row['comment'])
        
        # Regex to find @mentions
        mentions = re.findall(r'@(\w+)', text)
        for m in mentions:
            target = clean_username(m)
            if target in G.nodes() and target != user:
                if G.has_edge(user, target):
                    G[user][target]['weight'] += 2 # Higher weight for explicit mention
                else:
                    G.add_edge(user, target, weight=2)
                mention_count += 1
                
    # 4. Add Edges (Temporal Co-occurrence)
    print("Building Temporal Edges (this may take a moment)...")
    # Sort by time
    df_sorted = df.sort_values('timestamp')
    
    # Window-based co-occurrence (e.g., users commenting within 2 minutes)
    # To optimize: iterate and keep a buffer of recent comments
    temporal_edges = 0
    window_minutes = 2
    
    # Simple sliding window comparison (O(N*window_size))
    # converting to list of dicts for speed
    records = df_sorted[['username', 'timestamp']].to_dict('records')
    
    for i in range(len(records)):
        current = records[i]
        curr_time = current['timestamp']
        curr_user = current['username']
        
        # Look ahead
        for j in range(i + 1, len(records)):
            next_record = records[j]
            time_diff = (next_record['timestamp'] - curr_time).total_seconds() / 60.0
            
            if time_diff > window_minutes:
                break # Sorted, so we can stop early
                
            next_user = next_record['username']
            
            if curr_user != next_user:
                # Add temporal edge
                if G.has_edge(curr_user, next_user):
                    G[curr_user][next_user]['weight'] += 0.5 # Lower weight for temporal
                else:
                    G.add_edge(curr_user, next_user, weight=0.5)
                temporal_edges += 1

    print(f"Edges created: Mentions={mention_count}, Temporal={temporal_edges}")

    # 5. Community Detection
    print("Detecting Communities...")
    try:
        # Try finding communities
        communities = nx.community.greedy_modularity_communities(G)
        # Assign community ID
        for i, comm in enumerate(communities):
            for node in comm:
                G.nodes[node]['group'] = i
    except Exception as e:
        print(f"Community detection warning: {e}")
        for node in G.nodes():
            G.nodes[node]['group'] = 0

    # 6. Calculate Centrality (Potential Buzzer Indicator)
    print("Calculating Centrality...")
    degree_dict = dict(G.degree(weight='weight'))
    nx.set_node_attributes(G, degree_dict, 'degree')

    # 7. Export to JSON
    print(f"Exporting to {OUTPUT_JSON}...")
    
    # Format for standard visualization libraries (like react-force-graph)
    nodes_data = []
    for n, attr in G.nodes(data=True):
        nodes_data.append({
            "id": n,
            "label": n,
            "group": attr.get('group', 0),
            "size": attr.get('size', 1) * 3, # Scale for vis
            "sentiment": attr.get('sentiment', 'Netral'),
            "color": attr.get('color', 'gray'),
            "val": attr.get('degree', 1) # 'val' often used for node radius in 3d force graph
        })
        
    links_data = []
    for u, v, attr in G.edges(data=True):
        if attr['weight'] >= 1.0: # Filter weak links to reduce noise
            links_data.append({
                "source": u,
                "target": v,
                "value": attr['weight']
            })
            
    output_data = {
        "nodes": nodes_data,
        "links": links_data,
        "metadata": {
            "total_nodes": len(nodes_data),
            "total_links": len(links_data),
            "generated_at": str(pd.Timestamp.now())
        }
    }
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    main()
