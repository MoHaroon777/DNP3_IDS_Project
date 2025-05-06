\documentclass[a0paper,portrait]{baposter}

\usepackage{wrapfig}
\usepackage{lmodern}
\usepackage{lipsum,graphicx}
\usepackage[utf8]{inputenc} %unicode support
\usepackage[T1]{fontenc}

\selectcolormodel{cmyk}

\graphicspath{{figures/}} % Directory in which figures are stored

\newcommand{\compresslist}{%
\setlength{\itemsep}{0pt}%
\setlength{\parskip}{1pt}%
\setlength{\parsep}{0pt}%
}

\newenvironment{boenumerate}
  {\begin{enumerate}\renewcommand\labelenumi{\textbf\theenumi.}}
  {\end{enumerate}}


\begin{document}

\definecolor{Mycolor1}{HTML}{00FFFF}
\definecolor{Mycolor2}{HTML}{008080}

\begin{poster}
{
grid=false,
headerborder=open, % Adds a border around the header of content boxes
colspacing=1em, % Column spacing
bgColorOne=white, % Background color for the gradient on the left side of the poster
bgColorTwo=white, % Background color for the gradient on the right side of the poster
borderColor=Mycolor1, % Border color
headerColorOne=Mycolor2, % Background color for the header in the content boxes (left side)
headerColorTwo=Mycolor2, % Background color for the header in the content boxes (right side)
headerFontColor=white, % Text color for the header text in the content boxes
boxColorOne=white, % Background color of the content boxes
textborder=rounded, %rectangle, % Format of the border around content boxes, can be: none, bars, coils, triangles, rectangle, rounded, roundedsmall, roundedright or faded
eyecatcher=false, % Set to false for ignoring the left logo in the title and move the title left
headerheight=0.11\textheight, % Height of the header
headershape=rounded, % Specify the rounded corner in the content box headers, can be: rectangle, small-rounded, roundedright, roundedleft or rounded
headershade=plain,
headerfont=\Large\textsf, % Large, bold and sans serif font in the headers of content boxes
%textfont={\setlength{\parindent}{1.5em}}, % Uncomment for paragraph indentation
linewidth=2pt % Width of the border lines around content boxes
}
{}
%
%----------------------------------------------------------------------------------------
%	DNP3 Intrusion Detection System
%----------------------------------------------------------------------------------------
%
{
\textsf %Sans Serif
{
{DNP3 Intrusion Detection System}
}
} % Poster title
% {\vspace{0.2em} Add Author Name, Add another author name\\ 
% {\small \vspace{0.7em} Department of Computing, TU Dublin, Tallaght, Dublin, Ireland}} 
{\sf\vspace{0.2em}\\
Mohamud Haroon  % Author names
\vspace{0.1em}\\
\small{ School of Enterprise Computing and Digital Transformation, TU Dublin, Ireland
\vspace{0.2em}\\
X00145689@myTUDublin.ie  % Author email addresses
}
}
{\includegraphics[width=.25\linewidth]{TUDublin.jpg}} % TU Dublin logo


% this states the box starts at column 0 (edge of page), row 0 (top of page) for a span of 3 (columns wide)
\headerbox{Introduction}{name=introduction,column=0,row=0, span=3}{
Industrial Control Systems that regulate critical infrastructure frequently utilize the DNP3 protocol, which, due to its original design, has inherent security vulnerabilities. The vulnerability exposes critical services to cyberattacks, making it necessary for robust detection systems. In this research study, a bespoke Intrusion Detection System for DNP3 network traffic was developed from the open-source DNP3 Intrusion Detection Dataset of Radoglou-Grammatikis et al.
Initial tests with Artificial Neural Network frameworks were found to have limitations in effectively generalizing across the various DNP3 attack classes within this dataset. Accordingly, the research then moved on to compare and contrast traditional Machine Learning models like Random Forest, Gradient Boosting, XGBoost, and Support Vector Machine. A sophisticated preprocessing pipeline with data cleaning, scaling, imputation, and extreme dimensionality reduction via Principal Component Analysis retaining 15 components was employed.
Comparative analysis showed that the SVM model was the most promising. After hyperparameter tuning using GridSearchCV with an RBF kernel having C=50 and gamma=0.1, the final SVM model performed outstandingly on a completely unseen test set with 99.00\% accuracy and a 0.99 F1-score. Statistical tests also supported the SVM's significantly better performance compared to the other standard models tested. This paper concludes that an SVM classifier, supplemented with careful preprocessing and PCA, is a potent and statistically justified technique for intrusion detection in DNP3 traffic in this data set.

%\vspace{2cm} %remove this, only added for spacing

}

% this states the box starts at column 0 (edge of page), directly below the box labelled introduction for a span of 1 (column wide)
\headerbox{Dataset and Preprocessing}{name=subtopic1,column=0,below=introduction,span=1}{

%\vspace{0.15cm}
Employed the openly accessible "DNP3 Intrusion Detection Dataset" (Radoglou-Grammatikis et al., 2022) containing labelled normal and attack DNP3 network flows. Preprocessing comprised removal of irrelevant features, numerical encoding, median imputation for missing values, standard scaling, and significant dimensionality reduction via PCA to 15 components capturing >95\% variance. Divided the data into train, validation, and test sets for assessment.
}

% this states the box starts at column 0 (edge of page), directly below the box labelled subtopic1 for a span of 1 (column wide)

\headerbox{Research}{name=subtopic2,column=0,below=subtopic1,span=1}{
Deficiencies in widely used DNP3 protocol require robust Intrusion Detection Systems (IDS) to secure critical infrastructures. Machine Learning (ML) has the potential to do better than traditional IDS. In this project, ML approaches were compared, with initial shortcomings of Neural Networks on this specific DNP3 dataset, leading to a comparative study of traditional algorithms (RF, GB, XGB, SVM) after extensive preprocessing to identify the best approach.

\vspace{2cm} %remove this, only added for spacing

}

\headerbox{Model Development}{name=subtopic3,column=1,below=introduction,span=2}{
\vspace{0.3cm}

% inserts an image inside the box, 5 rows high
% \begin{wrapfigure}[5]{r}{0.25\textwidth}
% \centering
% \includegraphics[width=.9\linewidth]{dataset_img.png} 
% \end{wrapfigure}

\vspace{0.3cm}
\subsubsection*{1. Preprocessed Data :}
\begin{figure}[h!] % Added [h!] placement specifier - might need adjustment
\centering
\includegraphics[width=0.8\linewidth]{dataset_img.png} % Adjusted width slightly, ensure file exists
\caption*{The DNP3 Preprocessed Dataset (Sample)} % Changed caption slightly, removed numbering
\label{fig:Dataset}
\end{figure}
\vspace{0.1cm}
% \subsubsection*{1. Data Pre-processing Summary:}
% Before modeling, the DNP3 dataset underwent rigorous preprocessing. This included removing irrelevant features, handling missing values via median imputation, numerical encoding, standard scaling, and crucially, dimensionality reduction using PCA to 15 components, which captured over 95\% of the data variance. The data was split into training, validation, and test sets.


\vspace{0.1cm} 




% \begin{wrapfigure}[5]{r}{0.25\textwidth}
% \includegraphics[width=.9\linewidth]{cr_val_svm.png} 
% \end{wrapfigure}

\begin{minipage}[t]{0.45\textwidth}
    \vspace{0.3cm}
    Initial development explored Artificial Neural Networks (ANNs). However, the ANN models struggled to generalize effectively on this dataset. While the training accuracy increased, the validation accuracy quickly plateaued at a low level, indicating significant overfitting and an inability to learn meaningful patterns from the validation data.
\end{minipage}%
\hfill % This creates space between the two minipages
\begin{minipage}[t]{0.45\textwidth}
    \vspace{0.3cm}
    Due to these limitations of ANNs, the strategy pivoted to traditional Machine Learning algorithms. Comparative analysis revealed that Support Vector Machine (SVM) consistently outperformed Random Forest, Gradient Boosting, and XGBoost during preliminary evaluations on the validation set, making it the most promising candidate.
\end{minipage}


% \vspace{0.3cm}
% \subsubsection*{1. Data Pre-processing Summary:}
% Before modeling, the DNP3 dataset underwent rigorous preprocessing. This included removing irrelevant features, handling missing values via median imputation, numerical encoding, standard scaling, and crucially, dimensionality reduction using PCA to 15 components, which captured over 95\% of the data variance. The data was split into training, validation, and test sets.
% \vspace{0.3cm}
% \subsubsection*{2. Model Exploration  Comparison (ANN vs SVM):}
% Initial development explored Artificial Neural Networks (ANNs). However, as shown below, the ANN models struggled to generalize effectively on this dataset. The training accuracy increased, but validation accuracy quickly plateaued at a low level, indicating significant overfitting and an inability to learn meaningful patterns from the validation data.
% [PLACEHOLDER: Insert ANN Learning Curve Plot (Accuracy & Loss vs Epoch) Here]
% (Caption Suggestion: ANN Learning Curves showing limited validation performance and overfitting.)
% Due to these ANN limitations, the strategy pivoted to traditional Machine Learning algorithms. Comparative analysis revealed that Support Vector Machine (SVM) consistently outperformed Random Forest, Gradient Boosting, and XGBoost during preliminary evaluations on the validation set, making it the most promising candidate.
% \vspace{0.3cm}
% \subsubsection*{3. Final SVM Model Performance:}
% The SVM classifier was optimized using GridSearchCV (finding optimal C=50, gamma=0.1, kernel='rbf') and trained on the processed data. Evaluation on the separate validation and test sets yielded excellent results:
% Validation Set: Achieved 98.00\% accuracy with high precision, recall, and F1-scores across most classes (Macro F1: 0.98).
% Test Set (Unseen Data): Confirmed strong generalization with 99.00\% accuracy and equally high class-specific metrics (Macro F1: 0.99).


\headerbox{Model Results}{name=model_results,column=0,span=3,below=subtopic2}{
{
    {
        \hspace{1cm}
        \centering
    	\includegraphics[scale=.7]{svm_overall.png}
    }
    {
        \hspace{1.5cm}
        \centering
    	\includegraphics[scale=.5]{test_svm.png}
    }
}

}

\headerbox{Conclusions}{name=conclusion,column=0,below=topicoverview,span=2,above=bottom}{
It successfully applied a highly powerful Intrusion Detection System for the DNP3 protocol using a Support Vector Machine classifier in conjunction with PCA. The resulting model achieved 99.00\% accuracy and an F1-score of 0.99 for novel test instances, demonstrating high performance on all types of attacks. Statistical testing proved its effectiveness relative to other tested baseline ML models for this specific dataset. Although achieving high accuracy, SVM/PCA's use entails a compromise with model interpretability. Generally, the approach offers a solid, validated platform for enhancing DNP3 security.

\vspace{10cm} %remove this, only added for spacing

}

% \headerbox{QR Code for Recording}{name=conclusion,column=2,below=topicoverview,span=1,above=bottom}{
% QR Code Goes here

% \vspace{10cm} %remove this, only added for spacing

% }
\end{poster}

\end{document}
