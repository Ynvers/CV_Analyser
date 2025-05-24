import os
import base64
import streamlit as st
from dotenv import load_dotenv
from mistralai import Mistral


# Load environment variables
load_dotenv()
# Ensure the API key is set in the environment
if 'MISTRAL_API_KEY' not in os.environ:
    raise ValueError("MISTRAL_API_KEY environment variable is not set.")

# Initialize the Mistral client
client = Mistral(
    api_key=os.environ['MISTRAL_API_KEY'],
)

#Functions used in the app
def encode_file_from_stream(uploaded_file):
    """Encode the uploaded file to base64."""
    try:
        return base64.b64encode(uploaded_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {uploaded_file} was not found.")
        return None
    except Exception as e:  # Added general exception handling
        print(f"Error: {e}")
        return None

def to_ocr(uploaded_file):
    base64_pdf = encode_file_from_stream(uploaded_file)
    
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{base64_pdf}" 
        },
        include_image_base64=True
    )
    return ocr_response
    
system_prompt = """
Tu es un assistant expert en rédaction et optimisation de CV. Tu reçois le contenu textuel d’un CV (issu d’un OCR ou d’un PDF) et tu dois :
1-Identifier les informations clés : identité, titre professionnel, formations, expériences, compétences, langues, centres d’intérêts.
2-Vérifier la qualité de rédaction : orthographe, structure, clarté, lisibilité, présence des sections clés.
3-Évaluer la pertinence du contenu par rapport à au titre marqué
4-Proposer des conseils précis et personnalisés pour améliorer :
  -La forme (mise en page, hiérarchie des infos, mots d’action, lisibilité),
  -Le fond (mettre en valeur des expériences, reformuler un titre, ajouter des mots-clés du secteur, etc.),
  -La pertinence (cohérence avec un métier ou une offre visée).

Tu t’adresses à un utilisateur non-expert qui veut comprendre ce qu’il peut améliorer. Donne-lui :
-Une analyse résumée du CV (forces/faiblesses),
-Une liste de conseils actionnables (par points),
-Des suggestions de reformulations (titre, accroche, expériences si nécessaire),
-(Facultatif) Des ressources utiles (ex : lien vers des modèles de CV, explications de mots-clés à utiliser).

PS: Si certaines parties sont manquantes ou floues à cause de l’OCR, signale-le de manière claire et bienveillante.
"""

# Set up the Streamlit app
def analyse_chat():
    st.header("💬 Analyse and get some advice on your CV by our assistant")
    st.markdown("#### Paste the text of your CV below:")

    if "chat_history" not in st.session_state:
        # Initialize chat history with the system prompt
        st.session_state.chat_history = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": "Hello! I am here to help you analyze your CV. Bellow, is the first analysis of your CV:"
            ""},
        ]

        if st.session_state.ocr_text:
            # First call to the llm to generate an review of the CV
            user_prompt = f"Analyse the following CV text and provide feedback:\n\n{st.session_state.ocr_text}"
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})
            try:
                response = client.chat.complete(
                    model="mistral-medium-2505",
                    messages=st.session_state.chat_history,
                    temperature=0
                )
                reply = response.choices[0].message.content
            except Exception as e:
                st.error(f"⚠️ Error in Mistral API call: {e}")
                return
            
            # Append the assistant's response to the chat history
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

    for message in st.session_state.chat_history[1:]: #We forgot the system message
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("✍️ Ask a question or request advice on your CV:")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        # Call the Mistral API with the user input
        try:
            response = client.chat.complete(
                model="mistral-medium-2505",
                messages=st.session_state.chat_history,
                temperature=0
            )
            reply = response.choices[0].message.content
        except Exception as e:
            st.error(f"⚠️ Error in Mistral API call: {e}")
            return
        
        # Append the assistant's response to the chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response.choices[0].message.content})

        # Display the assistant's response
        with st.chat_message("assistant"):
            st.markdown(reply)

if "ocr_done" not in st.session_state:
    st.session_state.ocr_done = False
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""

def main():

    st.sidebar.title("⚙️ Options")
    if st.sidebar.button("🔄 Reboot la discussion"):
        st.session_state.chat_history = []
        st.session_state.ocr_done = False
        st.session_state.ocr_text = ""
        st.rerun()

    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    - This app uses the Mistral API to analyze and provide feedback on your CV.
    - It extracts text from uploaded documents using OCR and then generates a detailed analysis and suggestions for improvement.
    - The developer of this app is [Ynvers AKA Nathan the SpiderAI](https://github.com/Ynvers).               
    """)
    st.title("🧠 Mistral CV-Analyser")

    if not st.session_state.ocr_done:
        uploaded_file = st.file_uploader(
            "📄 Upload your CV here to have an analysis detailed", 
            type=["pdf", "jpg", "jpeg", "png"], 
            key="file_uploader"
        )

        if uploaded_file:
            with st.spinner("📚 Lecture et extraction du texte..."):
                cv_text = to_ocr(uploaded_file)

            if cv_text and hasattr(cv_text, "pages"):
                try:
                    # Vérification supplémentaire pour s'assurer que chaque page a un attribut markdown
                    ocr_text_content = "\n".join(
                        [page.markdown for page in cv_text.pages if hasattr(page, 'markdown')]
                    )
                    st.session_state.ocr_text = ocr_text_content
                    st.session_state.ocr_done = True
                    st.rerun()
                except Exception as e:
                    st.error(f"⚠️ Error in OCR's extraction : {e}")
            else:
                st.error("⚠️ Could not extract text from the document.")
    else:
        analyse_chat()


if __name__ == "__main__":
    main()