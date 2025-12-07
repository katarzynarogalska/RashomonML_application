
HR_dataset_dict = {
        "Title" : "Job Change of Data Scientists Dataset", 
        "Source": "https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists",
        "License": "CC0 Public Domain",
        "References" : "Author and Citations not specified",
        "Classification task": "binary classification",
        "Assigned category" : "Business application",
        "Short description" : "This dataset contains employee credentials, demographics, and experience to help analyze factors influencing job change or loyalty in the Data Science industry.",
        "Long description" : "A company active in Big Data and Data Science wants to hire data scientists from among people who successfully complete courses conducted by the company. Many people sign up for their training. The company wants to know which of these candidates truly want to work for the company after the training, and which are looking for new employment. This information helps to reduce the cost and time, as well as improve the quality of training or planning, course planning and candidate categorization. Information related to demographics, education and experience is available from candidates signup and enrollment.",
        "Target" : "Column name - target. 0 - Not looking for a job change, 1 - Looking for a job change",
        "Number of classes" : 2,
        "Number of records" : 19158,
        "Number of features" : 13,
        "Feature description" : [
            "enrollee_id: Unique ID for candidate",
            "city: City code",
            "city_ development _index: Development index of the city (scaled)",
            "gender: Gender of the candidate",
            "relevent_experience: Relevant experience of candidate",
            "enrolled_university: Type of University course enrolled if any",
            "education_level: Education level of candidate",
            "major_discipline: Education major discipline of candidate",
            "experience: Candidate total experience in years",
            "company_size: Number of employees in current employer's company",
            "company_type : Type of current employer",
            "last_new_job: Difference in years between previous job and current job",
            "training_hours: Training hours completed"
        ],
    "Note" : "Data for this dataset is the train data from source since test data did not have assigned target The train-test split was performed before training the models.",
    "Characteristics": "This dataset represents a real-life business use case. Companies may want to identify candidates genuinely interested in working there after training, which helps reduce costs, save time, and improve training quality and course planning. There is a great class imbalance, with the majority observations from class 0 - candidates who are not looking for a job change. Some features in the dataset exhibit skewed distributions, for example training_hours. Some information about the company is missing making up for around 8% of NaN values.",
    "path": "hr_job_change_ds.csv",
    "target_col" : "target"
}

credit_dict = {
    "Title" : "Credit Score Dataset", 
    "Source": "https://www.kaggle.com/datasets/sujithmandala/credit-score-classification-dataset",
    "License": "CC BY 4.0",
    "References" : "Author : Sujith K Mandala",
    "Classification task": "multiclass classification",
    "Assigned category" : "Business application",
    "Short description" : "This dataset contains personal information about individuals along with their credit score classification. It can be used to predict the likelihood of obtaining a loan based on a person’s demographic background.",
    "Long description" : "There are many factors that influence a client's ability to obtain a loan. Many people don't realize that personal and demographic characteristics - such as marital status or number of children - can affect their credit score just as much as their financial situation. This dataset illustrates the impact of these factors by classifying the likelihood of obtaining a loan as High, Average, or Low.",
    "Target" : "Column name - Credit Score",
    "Number of classes" : 3,
    "Number of records" : 164,
    "Number of features" : 7,
    "Feature description" : [
        "Age : person's age as a numeric value",
        "Gender : person's gender - Male or Female", 
        "Income : person's salary as a numeric value",
        "Education : category of the obtained diploma",
        "Marital status : persons marital status - Married or Single",
        "Number of Children : number of children as a numeric value",
        "Home ownership : category of home ownership - Owned or Rented",
        "Credit score : used as a target variable, credit score classified into categories such as High, Average, Low"
    ],
    "Note" : "Data from this dataset was saved from Kaggle in a form of one dataset, therefore it was splitted to train and test set during the preprocessing process",
    "Characteristics": "This dataset represents a real-life business use case. A financial institution can face challanges related to the predictive multiplicity while assigning the client's credit score.One model can assign a High credit score while the other will assign Low chances of getting a loan for the same person, who can be wrongfully harmed by the poor decision-making process. Other challenges that may arise from this dataset are class imbalance and feature correlation. Most observations were assigned the High credit score category, therefore ML algorithms may struggle to differentiate between Average and Low categories. Considering only numerical features, there is a high correlation between age and income present in the data, which can propose another challenge for the model to overcome.",
    "path" : "credit_score.csv",
    "target_col" : "Credit Score"
}


breast_cancer_dict = {
    "Title" : "Breast Cancer Diagnosis Dataset", 
    "Source": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
    "License": "CC BY-NC-SA 4.0",
    "References" : "Authors : William Wolberg, Olvi Mangasarian, Nick Street, W. Street. Citations - [K. P. Bennett, 'Decision Tree Construction Via Linear Programming.' Proceedings of the 4th Midwest Artificial Intelligence and Cognitive Science Society, pp. 97-101, 1992], [K. P. Bennett and O. L. Mangasarian: 'Robust Linear Programming Discrimination of Two Linearly Inseparable Sets', Optimization Methods and Software 1, 1992, 23-34]",
    "Classification task": "binary classification",
    "Assigned category" : "Predictive multiplicity problem",
    "Short description" : "This dataset consists of features related to the cell nuclei properties obtained from a digitized image of a fine needle aspirate (FNA) of a breast mass. The aim of this data is to classify the type of breast cancer present in the observed cells as malignant or benign.",
    "Long description" : " Modern cancer diagnosis allows medical staff to precisely analyze and assess abnormalities in the body through the examination of microscopic images. However, in many cases the human eye is not as observant as new computer algorithms, that’s why using new Machine Learning technologies to support the diagnostic process is becoming more and more popular. This dataset aims to classify abnormalities present in breast cells as malignant or benign types of cancer. An early diagnosis can increase the chances of successful treatment.",
    "Target" : "Column name - Diagnosis. M - malignant, B - benign",
    "Number of classes" : 2,
    "Number of records" : 569,
    "Number of features" : 32,
    "Feature description" : [
        "ID : id of an observation",
        "Diagnosis : used as target, M- malignant, B - benign",
        "Radius mean/se/worst : refer to the distances from center to points on the perimeter", 
        "Texture mean/se/worst : refer to the standard deviation of gray-scale values",
        "Perimeter mean/se/worst : refer to the size of the core tumor",
        "Area mean/se/worst : refer to the area of the cell nucleus", 
        "Smoothness mean/se/worst : refer to the local variation in radius lengths",
        "Compactness mean/se/worst : refer to mean of perimeter^2 / area - 1.0",
        "Concavity mean/se/worst : refer to the severity of concave portions of the contour",
        "Concave points mean/se/worst : refer to the number of concave portions of the contour",
        "Symmetry mean/se/worst : refer to the symmetry of the nucleus",
        "Fractal dimension mean/se/worst : refer to the 'coastline approximation' - 1",

    ],
    "Note" : "Data from this dataset was saved from Kaggle in a form of one dataset, therefore it was splitted to train and test set during the preprocessing process. Additionally target column was encoded to numerical values : M = 1 and B=0 for training purposes.",
    "Characteristics": "This dataset is a real life example of a situation when predictive multiplicity can directly affect a human's life. The lack of analysis of competing models and their conflicting predictions can lead to the overlooking and in consequence the lack of correct treatment for the patient. Although this data is not severely imbalanced (around 65-35 proportion), it has some highly correlated columns such as perimeter and radius mean, which can propose a new challenge for ML algorithms.",
    "path" : "breast_cancer.csv",
    "target_col" : "diagnosis"
}

heart_dict ={
    "Title" : "Heart Failure Dataset", 
    "Source": "https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data",
    "License": "Open Data Commons Open Database License (ODbL) v1.0",
    "References" : " Citations - fedesoriano. (September 2021). Heart Failure Prediction Dataset. Retrieved [Date Retrieved] from https://www.kaggle.com/fedesoriano/heart-failure-prediction",
    "Classification task" : "binary classification",
    "Assigned category" : "Predictive multiplicity problem",
    "Short description" :  "This dataset contains features for predicting cardiovascular disease, the leading cause of death worldwide",
    "Long description" :"Cardiovascular diseases are the number 1 cause of death globally. Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease. Early detection and management of cardiovascular diseases are crucial to prevent fatal outcomes. Machine learning models can help predict the presence of cardiovascular disease, but this also highlights the challenge of predictive multiplicity, which in some cases can lead to serious or even tragic consequences. That is why, experiments on this dataset can underline the problem of predictive multiplicity even more",
    "Target" : "Column name - HeartDisease. 1 - Heart disease, 0 - Normal",
    "Number of classes" : 2,
    "Number of records" : 918,
    "Number of features" : 11,
    "Feature description" : [
        "Age: age of the patient in years",
        "Sex: sex of the patient, M - male, F - female",
        "ChestPainType: chest pain type, TA - Typical Angina, ATA - Atypical Angina, NAP - Non-Anginal Pain, ASY - Asymptomatic",
        "RestingBP: resting blood pressure in mmHg",
        "Cholesterol: serum cholesterol in mm/dl",
        "FastingBS: fasting blood sugar, 1 - if FastingBS > 120 mg/dl, 0 - otherwise",
        "RestingECG: resting electrocardiogram results, Normal - Normal, ST - having ST-T wave abnormality, LVH - showing probable or definite left ventricular hypertrophy by Estes' criteria",
        "MaxHR: maximum heart rate achieved",
        "ExerciseAngina: exercise-induced angina, Y- yes, N - no",
        "Oldpeak: oldpeak = ST, numeric value measured in depression",
        "ST_Slope: the slope of the peak exercise ST segment, Up - upsloping, Flat - flat, Down - downsloping"


    ],
    "Note" : "Data from this dataset was saved from Kaggle in a form of one dataset, therefore it was splitted to train and test set during the preprocessing process.",
    "Characteristics": "This data is the representation or a real life situation when predictive multiplicity should be carefully examined before the decision-making process, as it may have tragic consequences. One model can predict that a person does not have heart disease, while another predicts that they do. If the individual is actually seriously ill but we rely on a model that predicts no disease, the misdiagnosis puts the person in serious risk. This dataset has well balanced classes and does not show signs of highly correlated features.",
    "path" : "heart.csv",
    "target_col" : "HeartDisease"

}

compas_dict ={
    "Title" : "COMPAS Recidivism Racial Bias Dataset", 
    "Source": "https://www.kaggle.com/datasets/danofer/compass/data",
    "License": "Database Contents License (DbCL) v1.0",
    "References" : "Authors - Propublica, Acknowledgements - Data & original analysis gathered by ProPublica.",
    "Classification task" : "binary classification",
    "Assigned category" : "Predictive multiplicity problem",
    "Short description" : "COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) is a risk assessment algorithm employed by judges and parole officers to estimate the probability of recidivism (reoffending) within a two-year period. A critical aspect in the context of predictive multiplicity is its known bias in favor of white defendants.",
    "Long description" :" COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) is a risk assessment algorithm employed by judges and parole officers to estimate the probability of recidivism (reoffending) within a two-year period. The research conducted by ProPublica stated: “Black defendants were often predicted to be at a higher risk of recidivism than they actually were. Our analysis found that black defendants who did not recidivate over a two-year period were nearly twice as likely to be misclassified as higher risk compared to their white counterparts (45 percent vs. 23 percent).White defendants were often predicted to be less risky than they were. Our analysis found that white defendants who re-offended within the next two years were mistakenly labeled low risk almost twice as often as black re-offenders (48 percent vs. 28 percent). The analysis also showed that even when controlling for prior crimes, future recidivism, age, and gender, black defendants were 45 percent more likely to be assigned higher risk scores than white defendants.”. This underscores a crucial concern in FairML, as the outcomes of algorithmic predictions can directly affect people’s lives. Since the algorithm is known to be biased, we are conducting our experiments on a dataset without the algorithm output to predict whether the defendant committed a crime in the next two-year period.",
    "Target" : "Column name - Two_yr_Recidivism, 0 - no recidivism , 1 - recidivism",
    "Number of classes" : 2,
    "Number of records" : 6172,
    "Number of features" : 11,
    "Feature description" : [
        "Number_of_Priors - number of prior offences defendant has committed",
        "Age_Above_FourtyFive - indicator if the defendant is above 45 years old",
        "Age_Below_TwentyFive -  indicator if defendant is below 25 years old",
        "African_American -  indicator if the defendant's race is above African American",
        "Asian - indicator if defendant’s race is above Asian",
        "Hispanic - indicator if defendant’s race is above Hispanic",
        "Native_American - indicator if defendant’s race is above Native American",
        "Other - indicator if defendant’s race is other than listed above",
        "Female - indicator if the defendant is a female",
        "Misdemeanor - indicator whether defendant has committed misdemeanor offense"
    ],
    "Note" : "Data from this dataset was saved from Kaggle in a form of one dataset, therefore it was splitted to train and test set during the preprocessing process. You can find this dataset at the source above, file used propublicaCompassRecividism_data_fairml.csv. The dataset contains fewer features, and all variables have been converted to numeric format to prevent data leakage (which occurs in other datasets). We dropped the Score_factor column since it is the output of the COMPAS algorithm which is biased in favor of white defendants.",
    "Characteristics": "This data is the representation or a real life situation when predictive multiplicity should be carefully examined before the decision-making process, as it may have harmful consequences. One model can predict that a defendant will commit a crime in the next two-year period, while another predicts that they will not. It is crucial for the model to not be biased in favor of any race so relying on model scores itself might be unfair to individuals. This dataset has well balanced classes and does not show signs of highly correlated features that would pose an additional challenge for ML algorithms.",
    "path" : "compas_violent.csv",
    "target_col" : "Two_yr_Recidivism"
}

glass_dict ={
    "Title" : "Glass Classification Dataset", 
    "Source": "https://www.kaggle.com/datasets/uciml/glass/data",
    "License": "Database Contents License (DbCL) v1.0",
    "References" : "German, B. (1987). Glass Identification [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5WW2P",
    "Classification task" : "multiclass classification",
    "Assigned category" : "Data challenging for ML algorithms",
    "Short description" : "This dataset contains information about six different types of glass, defined in terms of their oxide content.",
    "Long description" :" This dataset contains information on six different types of glass, defined in terms of their oxide content (i.e. Na - sodium, Fe - iron, K - potassium, etc.). Data was published by the USA Forensic Science Service in 1987 to predict the type of glass (from building windows and vehicle windows to containers and headlamps). It is worth noting that the original study of classification of types of glass was motivated by criminological investigation.",
    "Target" : "Column name - Type, 1 : building_windows_float_processed, 2 : building_windows_non_float_processed, 3 : vehicle_windows_float_processed, 5 containers, 6 tableware, 7 headlamps (class 4 is not present in the dataset) ",
    "Number of classes" : 6,
    "Number of records" : 214,
    "Number of features" : 9,
    "Feature description" : [
        "RI: refractive index",
        "Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)",
        "Mg: Magnesium",
        "Al: Aluminum",
        "Si: Silicon",
        "K: Potassium",
        "Ca: Calcium",
        "Ba: Barium",
        "Fe: Iron"

    ],
    "Note" : "Data from this dataset was saved from Kaggle in a form of one dataset, therefore it was splitted to train and test set during the preprocessing process.",
    "Characteristics" :"Features in this dataset are highly correlated, with the highest positive Pearson correlation value being 0.81 (between Ca and RI) and the strongest negative -0.74 value (between Mg and Type). This might be problematic for some machine learning models, especially linear models like linear regression. This dataset also shows class imbalance with most observations from classes 1 and 2.",
    "path" : "glass.csv",
    "target_col" : "Type"
}

letter_dict ={
    "Title" : "Letter Recognition Dataset", 
    "Source": "https://archive.ics.uci.edu/dataset/59/letter+recognition",
    "License": "CC by 4.0",
    "References" : "Author - David Slate",
    "Classification task" : "multiclass classification",
    "Assigned category" : "Data challenging for ML algorithms",
    "Short description" : "A dataset containing numerous pictures of letters converted to pixel characteristics allowing the ML algorithms to learn the stucture of the pixels and recognize the correct letter from the alphabet. ",
    "Long description" :"This dataset consists of rows represeting a single handwritten letter. Many different fonts and sizes of the letters pose an additional challange for ML algorithms. Each picture was firstly converted to pixel matrix with white pixels illustrating the letter, and the black ones being the background. Then raw pixels were converted into a set of numerical attributes such as height, width and total number of whit pixels. This dataset contains 26 possible classes, many of which occur very infequently. ",
    "Target" : "Column name - letter, capital letters from the English alphabet",
    "Number of classes" : 26,
    "Number of records" : 20000,
    "Number of features" : 16,
    "Feature description" : [
        "x-box : horizontal position of the box",
        "y-box : vertical position of the box",
        "width : width of the box",
        "height : height of the box",
        "onpix : total number of on pixels",
        "x-bar : mean x of on pixels in the box",
        "y-bar : mean y of on pixels in the box",
        "x2bar : mean x variance",
        "y2bar : mean y variance",
        "xybar : mean x y correlation",
        "x2ybr : mean of x*x*y",
        "xy2br : mean of x*y*y",
        "x-ege : mean edge count left to right",
        "xegvy : correlation of x-ege with y",
        "y-ege : mean edge count bottom to top",
        "yegvx : correlation of y-ege with x"
    ],
    "Note" :"Data from this dataset was saved from UCI in a form of one dataset, therefore it was splitted to train and test set during the preprocessing process. All feature descriptions were copied from the dataset source.",
    "Characteristics" : "This dataset is an example of a multiclassification dataset with a great number of classes, many of which occur very rarely. That is an additionall challenge for ML algorithms, especially to differ between the rare classes.",
    "path" : "letter_recognition.csv",
    "target_col" : "letter"
}

yeast_dict ={
    "Title" : "Yeast Dataset", 
    "Source": "https://archive.ics.uci.edu/dataset/110/yeast",
    "License": "CC by 4.0",
    "References" : "Author - Kenta Nakai, Citations related to the dataset - 'Expert System for Predicting Protein Localization Sites in Gram-Negative Bacteria', Kenta Nakai & Minoru Kanehisa, PROTEINS: Structure, Function, and Genetics 11:95-110, 1991, 'A Knowledge Base for Predicting Protein Localization Sites in Eukaryotic Cells', Kenta Nakai & Minoru Kanehisa, Genomics 14:897-911, 1992.",
    "Classification task" : "multiclass classification",
    "Assigned category" : "Data challenging for ML algorithms",
    "Short description" : "A dataset from molecular biology containing information about protein properties that allow predicting their cellular localization. ",
    "Long description" :"The dataset originates from research on yeast (Saccharomyces cerevisiae) proteins. It contains various features describing their psychochemical characteristics, which can be used to predict their subcellular localization. The possible protein localizations have been categorized into 10 classes such as for example mitochondria(MIT) or vacuole(VAC)",
    "Target" : "Column name - localization_site, MIT - Mitochondria, NUC - Nucleus, CYT - Cytoplasm, ME1/2/3 - Membrane protein site 1/2/3, EXC - Extracellular, VAC - Vacuole, POX - Peroxisome, ERL - Endoplasmic reticulum (lumen)",
    "Number of classes" : 10,
    "Number of records" : 1484,
    "Number of features" : 9,
    "Feature description" : [
        "Sequence Name : Accession number for the SWISS-PROT database",
        "Mcg : McGeoch's method for signal sequence recognition",
        "Gvh : von Heijne's method for signal sequence recognition",
        "Alm : Score of the ALOM membrane spanning region prediction program",
        "Mit : Score of discriminant analysis of the amino acid content of the N-terminal region (20 residues long) of mitochondrial and non-mitochondrial proteins",
        "Erl : Presence of HDEL substring (thought to act as a signal for retention in the endoplasmic reticulum lumen). Binary attribute",
        "Pox : Peroxisomal targeting signal in the C-terminus.",
        "Vac : Score of discriminant analysis of the amino acid content of vacuolar and extracellular proteins",
        "Nuc : Score of discriminant analysis of nuclear localization signals of nuclear and non-nuclear proteins",
        "Localization site : used as target, possible values are MIT - Mitochondria, NUC - Nucleus, CYT - Cytoplasm, ME1/2/3 - Membrane protein site 1/2/3, EXC - Extracellular, VAC - Vacuole, POX - Peroxisome, ERL - Endoplasmic reticulum (lumen)"

    ],
    "Note" :"Data from this dataset was saved from UCI in a form of one dataset, therefore it was splitted to train and test set during the preprocessing process. All feature descriptions were copied from the dataset source.",
    "Characteristics" :"This dataset may be particularly interesting as it is an example of a multiclass classification problem with multiple classes (26). In addition some features as width, height etc are highly correlated which can pose another challenge for ML algorithms",
    "path": "yeast.csv",
    "target_col" : "localization_site"
}

def return_datasets_descriptions():
    '''Returns the list of all pre-saved datasets dictionaries'''
    return [HR_dataset_dict, credit_dict, breast_cancer_dict, heart_dict, compas_dict, glass_dict, letter_dict, yeast_dict]