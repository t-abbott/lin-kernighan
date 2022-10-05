echo "compiling..."

pandoc notes.md \
    --toc \
    --pdf-engine=xelatex \
    -o output.pdf \
    -V papersize=A4 \
    -V geometry:margin=1in

echo "done"
