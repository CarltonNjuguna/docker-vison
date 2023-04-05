import streamlit as st
import pandas as pd
import cv2
from PIL import Image
import numpy as np
import os
import time
import sys
from ultralytics import YOLO
import shutil

def object_detection_image():
    st.title('Object Detection for Images')
    st.subheader("""
    This object detection project takes in an image and outputs the image with bounding boxes created around the objects in the image
    """)
    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Image originale", use_column_width=True)
        confThreshold =st.slider('Confidence', 0, 100, 50)
        nmsThreshold= st.slider('Threshold', 0, 100, 20)
        model = YOLO('best.pt')
        model.predict(source=image, conf=confThreshold/100,save=True)
        col1, col2 = st.columns(2)
        blur_image = Image.open("runs/detect/predict/image0.jpg").convert("RGB")
        gray_image = Image.open("runs/detect/predict/image0.jpg").convert("RGB")
        with col1:
            st.image(gray_image, caption="Image en niveaux de gris")
        with col2:
            st.image(blur_image, caption="Image avec effet de flou")
        shutil.rmtree('runs')

def yolo_seg():
    st.title('Yolov8 : Segmentation (sémantique ou par instance)')
    st.subheader("""
    Quels sont selon toi les meilleurs (et les plus époustouflants ?) jupyter notebook que l’on peut trouver à ce sujet sur le net concernant le computer vision à partir de datasets custom ?
    """)
    st.markdown("""To address this question, I will deviate slightly from the main topic, which concerns notebooks and datasets, or even the subject of YOLO. Instead, I would like to emphasize the importance of computer vision and, more broadly, artificial intelligence as a major revolution.
    It is fascinating to realize that many of the algorithms that underpin this revolution were conceived more than 20 years ago. It is only now that our machines and data allow us to fully exploit their potential, making the future even more promising.
    When discussing computer vision, one of the first examples that comes to mind is autonomous vehicles. The idea of being able to move around with the help of artificial intelligence alone seemed implausible not so long ago, and yet, it has become a reality.
    Computer vision encompasses a wide range of applications: security with the now-ubiquitous Face ID popularized by Apple, medicine with algorithms capable of detecting diseases invisible to the naked eye of a doctor, and machine learning models that help farmers detect an animal's illness from its excrement. On a larger scale, systems enable rapid defect detection in large factories.
    I am convinced that computer vision is only at the beginning of its revolution, and the changes it will bring will be beneficial for everyone.""")

def sys_embarq():
    st.title('Défis des systèmes embarqués')
    st.subheader("""
    Quels sont les défis auxquels le deep learning est confronté en termes de vitesse et de précision sur des systèmes embarqués?
    """)
    st.markdown("""
    Ressources limitées : Les systèmes embarqués ont généralement des ressources limitées en termes de puissance de calcul, de mémoire et de stockage. Cela rend difficile le déploiement de modèles de deep learning volumineux et complexes qui nécessitent beaucoup de ressources pour fonctionner efficacement.
    
    Consommation d'énergie : Les systèmes embarqués fonctionnent souvent sur des sources d'énergie limitées, comme des batteries. Les modèles de deep learning peuvent être gourmands en énergie, ce qui pose un problème pour les applications où l'autonomie énergétique est essentielle.
    
    Latence : Dans certaines applications en temps réel, les systèmes embarqués doivent traiter et analyser les données rapidement. Les modèles de deep learning complexes peuvent introduire une latence importante, rendant difficile leur utilisation dans des scénarios où la rapidité est cruciale.
    
    Précision : Réduire la taille des modèles pour les adapter aux contraintes des systèmes embarqués peut entraîner une perte de précision. Trouver le bon équilibre entre la taille du modèle et la précision est un défi important.

    """
    )
    st.subheader("""
    Quelles technologies ou frameworks peuvent être utilisés pour y remédier?
    """)
    st.markdown("""
    Modèles compressés : La compression des modèles, telle que la quantification et la factorisation matricielle peut réduire la taille du modèle et les ressources nécessaires pour l'exécution, tout en conservant une précision acceptable.
    
    Modèles optimisés pour le matériel : Concevoir des modèles de deep learning spécifiquement pour les systèmes embarqués en utilisant des architectures de réseau de neurones plus efficaces et moins gourmandes en ressources
    
    Frameworks spécialisés : Utiliser des frameworks optimisés pour les systèmes embarqués, tels que TensorFlow Lite et ONNX Runtime qui permettent de déployer facilement des modèles de deep learning sur des dispositifs à ressources limitées.
    
    Accélérateurs matériels : Utiliser des accélérateurs matériels spécialement conçus pour le deep learning, tels que les GPU et les TPU qui peuvent améliorer la vitesse d'exécution des modèles et réduire la consommation d'énergie.
    
    Réseaux neuronaux binaires (BNN) : Les BNN sont des réseaux de neurones dans lesquels les poids et les activations sont binarisés. Les BNN permettent de réduire considérablement la taille des modèles et la complexité des calculs, ce qui les rend particulièrement adaptés aux systèmes embarqués. Cependant, les BNN peuvent également entraîner une perte de précision par rapport aux modèles de deep learning traditionnels.
    """
    )
    st.subheader("""
    Avantage systèmes embarqués?
    """)
    st.markdown("""
    Latence réduite : Le traitement des données sur le dispositif lui-même permet de réduire la latence, car les données n'ont pas besoin d'être envoyées à un serveur distant pour être traitées.
    
    Confidentialité et sécurité des données : En traitant les données localement sur le dispositif, les systèmes embarqués peuvent améliorer la confidentialité et la sécurité des données, car elles ne sont pas transmises sur Internet. Cela est particulièrement important pour les applications qui gèrent des données sensibles, comme les informations médicales ou les données personnelles.
    
    Fonctionnement hors ligne : Les systèmes embarqués peuvent traiter les données et prendre des décisions en utilisant le deep learning même lorsqu'ils sont déconnectés d'Internet. Cela est particulièrement utile pour les applications fonctionnant dans des environnements où la connectivité réseau est limitée ou  inexistante, comme les drones, les satellites ou les véhicules sous-marins.
    
    Réduction de la bande passante et des coûts de communication : Le traitement des données sur le dispositif réduit la quantité de données qui doivent être transmises à un serveur distant, ce qui peut réduire la consommation de bande passante et les coûts de communication associés.
    
    Économie d'énergie : Le traitement des données localement sur un dispositif embarqué peut permettre d'économiser de l'énergie en réduisant les besoins en communication avec les serveurs distants
    """
    )

def main():
    new_title = '<p style="font-size: 42px;">Bienvenue à vous,PHOENIX AI!</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)

    read_me = st.markdown("""
    Dans le cadre de votre évaluation technique, voici l'application que j'ai développée :

    Cette application a été conçue à l'aide de Streamlit. Sur la gauche, vous trouverez un menu déroulant qui vous permet de sélectionner la section qui vous intéresse.
    
    De plus, voici les différents liens vers le code source, le jeu de données utilisé et le modèle :

    Code source : [insérer le lien]
    Jeu de données : [insérer le lien]
    Modèle : [insérer le lien]
    N'hésitez pas à les consulter pour mieux comprendre le fonctionnement de l'application.
    
    Amusez-vous bien !
    """
    )
    st.sidebar.title("Select Activity")
    choice  = st.sidebar.selectbox("MODE",("About","Object Detection(Image)","Défis des systèmes embarqués","Yolov8 : Segmentation"))
    
    if choice == "Object Detection(Image)":
        read_me_0.empty()
        read_me.empty()
        object_detection_image()
    elif choice == "Défis des systèmes embarqués":
        read_me_0.empty()
        read_me.empty()
        sys_embarq()
    elif choice == 'Yolov8 : Segmentation':
        read_me_0.empty()
        read_me.empty()
        yolo_seg()

    elif choice == "About":
        print()
        

if __name__ == '__main__':
		main()	