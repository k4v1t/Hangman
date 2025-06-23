import streamlit as st
from model.inference import load_model, predict_next_letter
import secrets

# ------------------------------
# Arcade Theme Styling with Retro Frame
# ------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

    /* Global font and color */
    * {
        font-family: 'Press Start 2P', monospace !important;
        font-size: 12px !important;
        color: #00ffff;
    }

    /* Set page-wide dark background */
    html, body, .stApp {
        background-color: #000 !important;
        background-repeat: repeat;
    }

    /* Darken Streamlit's content container too */
    .block-container {
        background-color: rgba(0, 0, 0, 0.85) !important;
    }

    /* CRT scanline overlay */
    body::before {
        content: "";
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        background: repeating-linear-gradient(
            to bottom,
            rgba(255,255,255,0.02),
            rgba(255,255,255,0.02) 2px,
            transparent 2px,
            transparent 4px
        );
        pointer-events: none;
        z-index: 9999;
    }

    /* Retro-style button */
    .stButton>button {
        background-color: #222;
        color: #0ff;
        border: 2px solid #0ff;
        border-radius: 0;
    }

    /* Retro-style input */
    .stTextInput>div>input {
        background-color: #000;
        color: #0ff;
    }

    /* Ensure all text elements follow the font */
    .stMarkdown, .stAlert, .stText, .stSubheader, .stCodeBlock {
        font-family: 'Press Start 2P', monospace !important;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# Game Intro Screen
# ------------------------------
if "game_started" not in st.session_state:

    st.title("🕹️ HangMAN vs AI")
    
    st.markdown("""
        Welcome to the retro showdown of the century!

        🧍‍♂️ **You** vs 🤖 **The Transformer**

        Each of you gets 6 lives. Take turns guessing letters to uncover the hidden word.

        💡 First to fully reveal the word — or reveal more of it before lives run out — wins. 
        
        If both sides reveal the word, the one with fewer wrong guesses wins!
                
        Click below to start the game!
    """)

    if st.button("🎮 Start Game"):
        st.session_state.game_started = True
        st.rerun()
    st.stop()

# ------------------------------
# Cached model loading
# ------------------------------
@st.cache_resource
def load_model_cached():
    load_model("hangman_vs_ai/model/transformer.pt")

if "model_loaded" not in st.session_state:
    with st.spinner("Loading AI model..."):
        load_model_cached()
    st.session_state.model_loaded = True
else:
    load_model_cached()

# ------------------------------
# Game Init
# ------------------------------
def get_random_word():
    with open("hangman_vs_ai/data/words.txt", "r") as f:
        words = [line.strip() for line in f if line.strip()]
    return secrets.choice(words)

def reset_game():
    keep_game_started = st.session_state.get("game_started", False)
    st.session_state.clear()
    st.session_state["game_started"] = keep_game_started

if "target_word" not in st.session_state:
    word = get_random_word()
    st.session_state.target_word = word
    st.session_state.masked_word = "_" * len(word)
    st.session_state.ai_masked_word = "_" * len(word)
    st.session_state.human_guessed = []
    st.session_state.human_wrong = 0
    st.session_state.ai_guessed = []
    st.session_state.ai_wrong = 0
    st.session_state.turn = "human"
    st.session_state.human_solved_on = 0
    st.session_state.ai_solved_on = 0
    st.session_state.human_done = False
    st.session_state.ai_done = False
    st.session_state.game_over = False
    st.session_state.outcome = ""

# ------------------------------
# Game State Display (Street Fighter Style)
# ------------------------------
col1, col_mid, col2 = st.columns([4, 1, 4])

with col1:
    
    st.image("hangman_vs_ai/assets/images/human_avatar.png", caption="PLAYER 1", width=150)

    health_left = 6 - st.session_state.human_wrong
    human_img_path = f"hangman_vs_ai/assets/healthbars/human_{health_left}_lives.png"
    st.image(human_img_path, use_container_width=True)

    correct_guesses = [g for g in st.session_state.human_guessed if g in st.session_state.target_word]
    wrong_guesses = [g for g in st.session_state.human_guessed if g not in st.session_state.target_word]
    st.markdown(f"❤️ Lives: {health_left}/6")
    st.markdown("✅ Correct: " + ", ".join(correct_guesses))
    st.markdown("❌ Wrong: " + ", ".join(wrong_guesses))
    st.markdown(f"**Word:** `{st.session_state.masked_word}`")

with col_mid:
    st.markdown("### VS")

with col2:
    
    st.image("hangman_vs_ai/assets/images/ai_avatar.png", caption="AI", width=150)  
    
    health_right = 6 - st.session_state.ai_wrong
    ai_img_path = f"hangman_vs_ai/assets/healthbars/ai_{health_right}_lives.png"
    st.image(ai_img_path, use_container_width=True)

    ai_correct = sum(1 for g in st.session_state.ai_guessed if g in st.session_state.target_word)
    ai_wrong = sum(1 for g in st.session_state.ai_guessed if g not in st.session_state.target_word)
    st.markdown(f"❤️ Lives: {health_right}/6")
    st.markdown(f"✅ Correct: {ai_correct}")
    st.markdown(f"❌ Wrong: {ai_wrong}")

# ------------------------------
# Turn + Game Over Checker
# ------------------------------
def check_turn_and_game_state():
    # Track completion status
    st.session_state.human_done = st.session_state.human_wrong >= 6 or "_" not in st.session_state.masked_word
    st.session_state.ai_done = st.session_state.ai_wrong >= 6 or "_" not in st.session_state.ai_masked_word

    # Initialize solved turn tracking
    if "turn_counter" not in st.session_state:
        st.session_state.turn_counter = 0
    st.session_state.turn_counter += 1

    if "human_solved_on" not in st.session_state and "_" not in st.session_state.masked_word:
        st.session_state.human_solved_on = st.session_state.turn_counter

    if "ai_solved_on" not in st.session_state and "_" not in st.session_state.ai_masked_word:
        st.session_state.ai_solved_on = st.session_state.turn_counter

    # If both players have solved or failed, compute the outcome
    if st.session_state.human_done and st.session_state.ai_done:
        st.session_state.game_over = True

        human_solved = "_" not in st.session_state.masked_word
        ai_solved = "_" not in st.session_state.ai_masked_word

        if human_solved and ai_solved:
            if st.session_state.human_solved_on < st.session_state.ai_solved_on:
                outcome = "🎉 You revealed the word first! You win!"
            elif st.session_state.ai_solved_on < st.session_state.human_solved_on:
                outcome = "🤖 AI revealed the word first! AI wins!"
            else:
                # Both solved on same turn — compare wrong guesses
                if st.session_state.human_wrong < st.session_state.ai_wrong:
                    outcome = "🎉 You both solved the word, but you made fewer mistakes. You win!"
                elif st.session_state.ai_wrong < st.session_state.human_wrong:
                    outcome = "🤖 You both solved the word, but the AI made fewer mistakes. AI wins!"
                else:
                    outcome = "🤝 It's a tie! Equal guesses and mistakes."

        elif not human_solved and not ai_solved:
            human_revealed = len(st.session_state.masked_word) - st.session_state.masked_word.count('_')
            ai_revealed = len(st.session_state.ai_masked_word) - st.session_state.ai_masked_word.count('_')
            if human_revealed > ai_revealed:
                outcome = "📊 Both out of lives. You revealed more of the word. You win!"
            elif ai_revealed > human_revealed:
                outcome = "🤖 Both out of lives. AI revealed more of the word. AI wins!"
            else:
                outcome = "🤷 Both out of lives. It's a tie!"

        elif human_solved:
            outcome = "🎉 You revealed the full word! You win!"
        else:
            outcome = "🤖 AI revealed the full word! AI wins!"

        st.session_state.outcome = outcome
        return

    # If one side finishes, allow the other to take final turn
    if "_" not in st.session_state.masked_word and not st.session_state.ai_done:
        st.session_state.turn = "ai"
        return

    if "_" not in st.session_state.ai_masked_word and not st.session_state.human_done:
        st.session_state.turn = "human"
        return

    # If one is out of lives, let the other keep going
    if st.session_state.human_done and not st.session_state.ai_done:
        st.session_state.turn = "ai"
        return

    if st.session_state.ai_done and not st.session_state.human_done:
        st.session_state.turn = "human"
        return

    # Switch turns normally
    if st.session_state.turn == "human":
        st.session_state.turn = "ai"
    else:
        st.session_state.turn = "human"

# ------------------------------
# Game Over Display
# ------------------------------
if st.session_state.game_over:
    st.markdown("---")
    st.subheader("🧠 Game Over")
    st.write(st.session_state.outcome)
    st.markdown(f"**The word was:** `{st.session_state.target_word}`")
    st.button("Play Again", on_click=lambda: reset_game())

# ------------------------------
# Human Guess + AI Guess
# ------------------------------
if not st.session_state.game_over and st.session_state.turn == "human":
    if st.session_state.get("clear_input", False):
        st.session_state["guess_input"] = ""
        st.session_state["clear_input"] = False

    guess = st.text_input(
        label="Your letter guess (a-z)",
        max_chars=1,
        key="guess_input",
        label_visibility="collapsed",
        placeholder="Type a letter..."
    )

    st.markdown("""
        <script>
        setTimeout(() => {
            const input = window.document.querySelector('input[data-testid="stTextInput"]');
            if (input) input.focus();
        }, 300);
        </script>
    """, unsafe_allow_html=True)

    if guess:
        guess = guess.lower()

        if not guess.isalpha():
            st.warning("🚫 Please enter a single alphabetical letter (a–z).")

        elif guess in st.session_state.human_guessed:
            st.warning(f"❗ You've already guessed '{guess}'. Try a new letter.")

        else:
            st.session_state.human_guessed.append(guess)

            if guess in st.session_state.target_word:
                new_mask = list(st.session_state.masked_word)
                for i, c in enumerate(st.session_state.target_word):
                    if c == guess:
                        new_mask[i] = guess
                st.session_state.masked_word = "".join(new_mask)
            else:
                st.session_state.human_wrong += 1

            st.session_state.clear_input = True
            check_turn_and_game_state()
            st.rerun()

elif not st.session_state.game_over and st.session_state.turn == "ai":

    ai_guess = predict_next_letter(
        current_state=st.session_state.ai_masked_word,
        guessed_letters=st.session_state.ai_guessed
    )

    if ai_guess not in st.session_state.ai_guessed:
        st.session_state.ai_guessed.append(ai_guess)

        if ai_guess in st.session_state.target_word:
            new_ai_mask = list(st.session_state.ai_masked_word)
            for i, c in enumerate(st.session_state.target_word):
                if c == ai_guess:
                    new_ai_mask[i] = ai_guess
            st.session_state.ai_masked_word = "".join(new_ai_mask)
        else:
            st.session_state.ai_wrong += 1

    check_turn_and_game_state()
    st.rerun()


st.markdown("""
<hr style="margin-top: 3em; border: none; border-top: 2px dashed #0ff;" />
<p>Want to team up IRL? Check these out:</p>
<div style='text-align: center; padding-bottom: 1em; font-size: 10px;'>
    <a href='https://github.com/k4v1t' target='_blank' style='text-decoration: none; margin: 0 20px; color: #00ffff;'>🕹 GitHub</a>
    <a href='https://www.linkedin.com/in/kavittolia' target='_blank' style='text-decoration: none; margin: 0 20px; color: #00ffff;'>💼 LinkedIn</a>
    <a href='https://raw.githubusercontent.com/k4v1t/CV/main/CV_Kavit_Tolia.pdf' target='_blank' style='text-decoration: none; margin: 0 20px; color: #00ffff;'>📄 CV</a>
</div>
""", unsafe_allow_html=True)