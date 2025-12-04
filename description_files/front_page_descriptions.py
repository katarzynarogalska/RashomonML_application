
front_page_overview = """ This website allows you to explore machine learning models that perform similarly well - their differences and similarities.
Analyze how models differ in their predictions, how consistent they are, and what trade-offs exist on the chosen dataset. 
Discover alternative solutions to ensure the right, fair and unbiased desicion making. 
"""

package_overview = """
This application is built on the Python library RashoML, which allows users to analyze sets of models trained for a particular classification problem.
It was developed to enable experimentation with the library's features in a no-code environment through an interactive interface and pre-saved datasets. 
All available datasets and their characteristics are described on the <strong> Datasets page</strong>. The application consists of two analytical dashboards : the <strong> Rashomon page </strong> and the <strong>Intersection page</strong>. 
Detailed explanations of key concepts, such as the Rashomon Set, the Rashomon Intersection, and all related metrics, are provided in the sections below to help users better understand the analysis.  

"""

rashomon_set_definition ="""
To understand the concept of the Rashomon Set, let's consider the situation illustrated by the diagram on the left. In many real-world cases, decision-makers train multiple models,evaluate their performance, and base their decisions on the predictions of the best-performing model.
However, as the diagram shows, there are often many models with similarly strong performance. For example, Model 1 might achieve an accuracy of 0.99, while Model 2 is only 1% lower.
<strong>The Rashomon Set</strong> consists of all models whose performance is within a small margin (epsilon) of the best model. 
Among them, the model with the highest score is called the <strong>base model</strong>, and the others are referred to as <strong>competing models</strong>.

"""

rashomon_set_situation="""
Regarding the illustration, the best model predicted the negative class for the client number 10. Depending on the classification problem, this could mean that the client will not receive a loan or insurance, or that they are not a carrier of a disease.
Upon closer examination of this situation, we notice that Model 2 is is part of the Rashomon Set if we allow an error margin of 1%. This second model, while achieving similarly strong overall performance, gives a different prediction for client number 10. 
This raises an important question : <strong>how many individual predictions would change if we chose the competing model over the best model?</strong> Perhaps, Model 2 is more interpretable, more stable and might even be preferable to Model 1. <br> <br>
All of these considerations should be carefully analyzed before making any final decisions, especially when these decisions have a direct impact on people's lives, to ensure that each decision can be clearly explained and that individuals can trust that their prediction is final and well-founded. 

"""
rashomon_metrics="""
 • <strong>Rashomon Ratio</strong> - the proportion of models that are included in the Rashomon Set from all trained models <br>
 • <strong>Rashomon Pattern Ratio</strong> - ...
"""


intersection_definition =""

