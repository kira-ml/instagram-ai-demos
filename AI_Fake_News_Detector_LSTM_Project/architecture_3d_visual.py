"""
3D Interactive Architecture Visualization for Fake News Detector
Uses Plotly for 3D animated/interactive neural network visualization
Accurate to the actual CNN-LSTM architecture in fake_news_detector_reel.py

Architecture from fake_news_detector_reel.py:
1. Input Layer: (None, 20) - sequence_length
2. Embedding Layer: vocab_size=5000, embedding_dim=100
3. Conv1D: filters=64, kernel_size=3, activation='relu'
4. MaxPooling1D: pool_size=2
5. LSTM: units=64
6. Dense: units=32, activation='relu'
7. Dropout: rate=0.3
8. Dense: units=16, activation='relu'
9. Output: units=1, activation='sigmoid'
"""

import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import os

# Architecture configuration (from fake_news_detector_reel.py)
ARCHITECTURE = {
    'max_sequence_length': 20,
    'vocab_size': 5000,
    'embedding_dim': 100,
    'cnn_filters': 64,
    'cnn_kernel_size': 3,
    'lstm_units': 64,
    'dense_units': 32,
    'dropout_rate': 0.3,
}

def create_3d_network_architecture(save_html: bool = True, save_image: bool = True):
    """
    Create an interactive 3D visualization of the neural network architecture.
    Each layer is represented as a 3D shape with accurate dimensions.
    """
    
    fig = go.Figure()
    
    # Layer definitions with positions and colors
    layers = [
        {
            'name': 'INPUT',
            'shape': 'Text Sequence',
            'size': f'{ARCHITECTURE["max_sequence_length"]} tokens',
            'z': 0,
            'color': 'rgb(255, 107, 157)',
            'width': 2,
            'height': 0.5,
            'depth': 0.5
        },
        {
            'name': 'EMBEDDING',
            'shape': f'Dense Matrix',
            'size': f'{ARCHITECTURE["vocab_size"]} × {ARCHITECTURE["embedding_dim"]}',
            'z': 1.5,
            'color': 'rgb(196, 69, 105)',
            'width': 3,
            'height': 1.2,
            'depth': 0.8
        },
        {
            'name': 'CNN (Conv1D)',
            'shape': 'Feature Maps',
            'size': f'{ARCHITECTURE["cnn_filters"]} filters, kernel={ARCHITECTURE["cnn_kernel_size"]}',
            'z': 3.5,
            'color': 'rgb(249, 127, 81)',
            'width': 3.5,
            'height': 1.5,
            'depth': 1.0
        },
        {
            'name': 'MaxPooling',
            'shape': 'Downsampling',
            'size': 'pool_size=2',
            'z': 5,
            'color': 'rgb(254, 202, 87)',
            'width': 2.5,
            'height': 1.0,
            'depth': 0.7
        },
        {
            'name': 'LSTM',
            'shape': 'Recurrent Layer',
            'size': f'{ARCHITECTURE["lstm_units"]} units',
            'z': 6.8,
            'color': 'rgb(72, 219, 251)',
            'width': 4,
            'height': 1.8,
            'depth': 1.2
        },
        {
            'name': f'Dense ({ARCHITECTURE["dense_units"]})',
            'shape': 'Fully Connected',
            'size': f'{ARCHITECTURE["dense_units"]} neurons + ReLU',
            'z': 9,
            'color': 'rgb(10, 189, 227)',
            'width': 2.5,
            'height': 1.0,
            'depth': 0.8
        },
        {
            'name': f'Dropout',
            'shape': 'Regularization',
            'size': f'rate={ARCHITECTURE["dropout_rate"]}',
            'z': 10.5,
            'color': 'rgb(139, 148, 158)',
            'width': 2.5,
            'height': 0.8,
            'depth': 0.6
        },
        {
            'name': f'Dense ({ARCHITECTURE["dense_units"]//2})',
            'shape': 'Fully Connected',
            'size': f'{ARCHITECTURE["dense_units"]//2} neurons + ReLU',
            'z': 12,
            'color': 'rgb(10, 189, 227)',
            'width': 2.0,
            'height': 0.8,
            'depth': 0.6
        },
        {
            'name': 'OUTPUT',
            'shape': 'Binary Classification',
            'size': '1 unit (sigmoid)',
            'z': 13.5,
            'color': 'rgb(0, 210, 211)',
            'width': 1.5,
            'height': 0.5,
            'depth': 0.5
        }
    ]
    
    # Create 3D boxes for each layer
    for layer in layers:
        # Create a 3D box (cuboid) for each layer
        x_center, y_center = 0, 0
        w, h, d = layer['width'], layer['height'], layer['depth']
        z = layer['z']
        
        # Define vertices of the box
        vertices = np.array([
            [-w/2, -h/2, z - d/2],
            [w/2, -h/2, z - d/2],
            [w/2, h/2, z - d/2],
            [-w/2, h/2, z - d/2],
            [-w/2, -h/2, z + d/2],
            [w/2, -h/2, z + d/2],
            [w/2, h/2, z + d/2],
            [-w/2, h/2, z + d/2]
        ])
        
        # Define faces (each face has 4 vertices)
        faces = [
            [0, 1, 2, 3],  # bottom
            [4, 5, 6, 7],  # top
            [0, 1, 5, 4],  # front
            [2, 3, 7, 6],  # back
            [0, 3, 7, 4],  # left
            [1, 2, 6, 5]   # right
        ]
        
        # Add layer box using Mesh3d
        for face in faces:
            face_vertices = vertices[face]
            fig.add_trace(go.Mesh3d(
                x=face_vertices[:, 0],
                y=face_vertices[:, 1],
                z=face_vertices[:, 2],
                color=layer['color'],
                opacity=0.7,
                name=layer['name'],
                showlegend=False,
                hovertemplate=f"<b>{layer['name']}</b><br>" +
                              f"Shape: {layer['shape']}<br>" +
                              f"Size: {layer['size']}<extra></extra>"
            ))
        
        # Add text annotation for layer name
        fig.add_trace(go.Scatter3d(
            x=[x_center + w/2 + 1],
            y=[y_center],
            z=[z],
            mode='text',
            text=[f"<b>{layer['name']}</b><br>{layer['size']}"],
            textfont=dict(size=12, color=layer['color']),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add connecting lines between layers
        if z > 0:
            prev_z = layers[layers.index(layer) - 1]['z']
            fig.add_trace(go.Scatter3d(
                x=[0, 0],
                y=[0, 0],
                z=[prev_z + layers[layers.index(layer) - 1]['depth']/2, z - d/2],
                mode='lines',
                line=dict(color='rgba(255, 255, 255, 0.3)', width=4),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Add title and layout configuration
    fig.update_layout(
        title={
            'text': '🧠 3D Neural Network Architecture<br>Fake News Detector (CNN-LSTM)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': 'white'}
        },
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title='', showbackground=False),
            yaxis=dict(showgrid=False, showticklabels=False, title='', showbackground=False),
            zaxis=dict(showgrid=True, title='Network Depth', gridcolor='rgba(255,255,255,0.1)'),
            bgcolor='rgb(20, 20, 20)',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=0.8),
                center=dict(x=0, y=0, z=0)
            )
        ),
        paper_bgcolor='rgb(20, 20, 20)',
        plot_bgcolor='rgb(20, 20, 20)',
        font=dict(color='white'),
        showlegend=False,
        width=1080,
        height=1080,
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    # Save outputs
    if save_html:
        output_html = "architecture_3d_interactive.html"
        fig.write_html(output_html)
        print(f"✅ Interactive 3D visualization saved: {output_html}")
        print(f"   Open in browser to rotate, zoom, and explore!")
    
    if save_image:
        output_png = "architecture_3d_static.png"
        try:
            fig.write_image(output_png, width=1080, height=1080)
            print(f"✅ Static 3D image saved: {output_png}")
        except Exception as e:
            print(f"⚠️  Could not save PNG (install kaleido: pip install kaleido)")
            print(f"   Error: {e}")
    
    return fig


def create_data_flow_3d(save_html: bool = True):
    """
    Create a 3D data flow visualization showing how data transforms
    through each layer with actual dimensions.
    """
    
    fig = go.Figure()
    
    # Data transformation stages with actual shapes
    stages = [
        {'name': 'Input Text', 'shape': (20,), 'z': 0, 'color': 'rgb(255, 107, 157)'},
        {'name': 'Embeddings', 'shape': (20, 100), 'z': 2, 'color': 'rgb(196, 69, 105)'},
        {'name': 'CNN Output', 'shape': (10, 64), 'z': 4, 'color': 'rgb(249, 127, 81)'},
        {'name': 'LSTM Output', 'shape': (64,), 'z': 6, 'color': 'rgb(72, 219, 251)'},
        {'name': 'Dense Layer', 'shape': (32,), 'z': 8, 'color': 'rgb(10, 189, 227)'},
        {'name': 'Prediction', 'shape': (1,), 'z': 10, 'color': 'rgb(0, 210, 211)'}
    ]
    
    for stage in stages:
        shape = stage['shape']
        z = stage['z']
        
        # Determine visualization based on shape dimensions
        if len(shape) == 1:
            # 1D vector - show as tall column
            width = 0.5
            height = shape[0] / 10
        elif len(shape) == 2:
            # 2D matrix - show as flat surface
            width = shape[1] / 30
            height = shape[0] / 10
        
        # Create 3D representation
        vertices = np.array([
            [-width, -height, z - 0.2],
            [width, -height, z - 0.2],
            [width, height, z - 0.2],
            [-width, height, z - 0.2],
            [-width, -height, z + 0.2],
            [width, -height, z + 0.2],
            [width, height, z + 0.2],
            [-width, height, z + 0.2]
        ])
        
        # Create mesh for the stage
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            color=stage['color'],
            opacity=0.8,
            name=stage['name'],
            hovertemplate=f"<b>{stage['name']}</b><br>Shape: {shape}<extra></extra>"
        ))
        
        # Add label
        fig.add_trace(go.Scatter3d(
            x=[width + 1.5],
            y=[0],
            z=[z],
            mode='text',
            text=[f"<b>{stage['name']}</b><br>{shape}"],
            textfont=dict(size=11, color=stage['color']),
            showlegend=False
        ))
    
    # Configure layout
    fig.update_layout(
        title={
            'text': '📊 3D Data Flow Transformation<br>Input → Prediction',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22, 'color': 'white'}
        },
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, showbackground=False),
            yaxis=dict(showgrid=False, showticklabels=False, showbackground=False),
            zaxis=dict(showgrid=True, title='Processing Flow', gridcolor='rgba(255,255,255,0.1)'),
            bgcolor='rgb(20, 20, 20)',
            camera=dict(eye=dict(x=2, y=2, z=1))
        ),
        paper_bgcolor='rgb(20, 20, 20)',
        font=dict(color='white'),
        showlegend=False,
        width=1080,
        height=1080,
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    if save_html:
        output = "data_flow_3d_interactive.html"
        fig.write_html(output)
        print(f"✅ 3D Data Flow visualization saved: {output}")
    
    return fig


def create_rotating_animation(output_file: str = "architecture_3d_rotating.html"):
    """
    Create an auto-rotating 3D animation perfect for Instagram Stories/Reels.
    This creates a full 360° rotation animation.
    """
    
    fig = create_3d_network_architecture(save_html=False, save_image=False)
    
    # Create frames for rotation animation (360 degrees)
    frames = []
    n_frames = 120  # 120 frames for smooth rotation
    
    for i in range(n_frames):
        angle = i * (360 / n_frames)
        rad = np.radians(angle)
        
        # Calculate camera position for circular rotation
        x = 1.8 * np.cos(rad)
        y = 1.8 * np.sin(rad)
        z = 0.8
        
        frame = go.Frame(
            layout=dict(
                scene=dict(
                    camera=dict(
                        eye=dict(x=x, y=y, z=z),
                        center=dict(x=0, y=0, z=0)
                    )
                )
            ),
            name=str(i)
        )
        frames.append(frame)
    
    fig.frames = frames
    
    # Add animation controls
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(label='▶ Play',
                         method='animate',
                         args=[None, dict(frame=dict(duration=50, redraw=True),
                                        fromcurrent=True,
                                        mode='immediate')]),
                    dict(label='⏸ Pause',
                         method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                          mode='immediate')])
                ],
                x=0.1,
                y=0.95
            )
        ],
        sliders=[dict(
            active=0,
            steps=[dict(args=[[f.name], dict(frame=dict(duration=0, redraw=True),
                                            mode='immediate')],
                       method='animate',
                       label=str(k))
                  for k, f in enumerate(fig.frames)],
            x=0.1,
            y=0.05,
            len=0.8
        )]
    )
    
    fig.write_html(output_file)
    print(f"✅ Rotating animation saved: {output_file}")
    print(f"   Perfect for Instagram Reels - auto-rotates 360°!")
    
    return fig


if __name__ == "__main__":
    print("🎨 Creating 3D Architecture Visualizations...")
    print("=" * 60)
    print("\n📐 Architecture Configuration:")
    for key, value in ARCHITECTURE.items():
        print(f"   {key}: {value}")
    print("\n" + "=" * 60)
    
    # Create all visualizations
    print("\n1️⃣  Creating interactive 3D network architecture...")
    create_3d_network_architecture(save_html=True, save_image=True)
    
    print("\n2️⃣  Creating 3D data flow visualization...")
    create_data_flow_3d(save_html=True)
    
    print("\n3️⃣  Creating rotating animation for Instagram...")
    create_rotating_animation()
    
    print("\n" + "=" * 60)
    print("✨ All 3D visualizations created!")
    print("\n📁 Output files:")
    print("   • architecture_3d_interactive.html (rotate & explore)")
    print("   • architecture_3d_static.png (static image)")
    print("   • data_flow_3d_interactive.html (data transformation)")
    print("   • architecture_3d_rotating.html (auto-rotating animation)")
    print("\n💡 Tips:")
    print("   • Open .html files in browser to interact")
    print("   • Use rotating animation for Instagram Reels")
    print("   • Screen record the HTML for video content")
    print("=" * 60)
