from graphviz import Digraph

def create_maple_workflow():
    # Create a new directed graph
    dot = Digraph(comment='MAPLE Workflow')
    dot.attr(rankdir='TB')
    
    # Set node styles
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
    
    # Create nodes for each stage
    # LLM Heuristics Stage
    dot.node('llm', 'LLM Heuristics\n(Claude 3.7)')
    
    # Data Synthesis Stage
    dot.node('data_synth', 'Data Synthesis\n(Best Algorithm Selection)')
    
    # Hardcoding Stage
    dot.node('hardcode', 'Algorithm Hardcoding')
    
    # Plan Generation Stage
    dot.node('plan_gen', 'Plan Generation')
    
    # Validation Stage
    dot.node('validation', 'Plan Validation\n(Validity & Optimality)')
    
    # Evaluation Stage
    dot.node('eval', 'Dataset Evaluation\n(Existing & New)')
    
    # Global Replanning Stage
    dot.node('replan', 'Global Replanning\n(Disruption Handling)')
    
    # Add edges to show workflow
    dot.edge('llm', 'data_synth')
    dot.edge('data_synth', 'hardcode')
    dot.edge('hardcode', 'plan_gen')
    dot.edge('plan_gen', 'validation')
    dot.edge('validation', 'eval')
    dot.edge('eval', 'replan')
    
    # Add feedback loops
    dot.edge('validation', 'plan_gen', 'Invalid Plan')
    dot.edge('eval', 'plan_gen', 'Poor Performance')
    dot.edge('replan', 'plan_gen', 'Disruption')
    
    # Save the diagram
    dot.render('maple_workflow', format='png', cleanup=True)
    print("Workflow diagram generated as 'maple_workflow.png'")

if __name__ == '__main__':
    create_maple_workflow() 