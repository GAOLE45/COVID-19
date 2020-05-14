mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = ture\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml