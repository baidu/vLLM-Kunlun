## 🚀 Installation

```bash

uv venv myenv --python 3.12 --seed
source myenv/bin/activate


 # Step 1: Enter the docs directory
cd docs

# Step 2: Install dependencies (using uv)
uv pip install -r requirements-docs.txt

# Install sphinx-autobuild (if not in requirements file)
uv pip install sphinx-autobuild

# Run from the docs directory:
sphinx-autobuild ./source ./_build/html --port 8000

# Step 1: Clean up old files
make clean

# Step 2: Build HTML
make html

# Step 3: Local preview
python -m http.server -d _build/html/

Browser access: http://localhost:8000

🌍 Internationalization
Internationalization translation process (taking Chinese as an example)

# Step 1: Extract translatable text (generate .pot)
sphinx-build -b gettext source _build/gettext

# Step 2: Generate/update Chinese .po file
sphinx-intl update -p _build/gettext -l zh_CN

# Step 3: Manually translate .po file
# Use a text editor to open source/locale/zh_CN/LC_MESSAGES/*.po
# Fill in the Chinese translation in msgstr ""

# Step 4: Compile and build Chinese documentation
make intl

# Step 5: View the effect
python -m http.server -d _build/html


Browser access:

English version: http://localhost:8000
Chinese version: http://localhost:8000/zh-cn

```
