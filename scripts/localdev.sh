
 #!/bin/bash  

: '
Script to run app w/o Docker.

* If you need to alter permissions to run script, chmod +x <file-name>.sh
'

# Run app
poetry run streamlit run app/main.py