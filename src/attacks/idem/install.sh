mkdir -p data/surrogate_models
cd data/surrogate_models

# Download the surrogate models
gdown --fuzzy  https://drive.google.com/file/d/1WR7ZARWrZJZParHL1s_S-t6X-mzBwBhd/view\?usp\=sharing -O s1.tar.gz
gdown --fuzzy  https://drive.google.com/file/d/1R176s7NKLy6UQHwQxMJCHoG_AVrmYCHn/view -O s2.tar.gz
gdown --fuzzy  https://drive.google.com/file/d/1w3y19XnwfqZkV3ELy5KEaralzqI1wbK6/view -O s3.tar.gz

# Extract the surrogate models
tar -xvf s1.tar.gz
tar -xvf s2.tar.gz
tar -xvf s3.tar.gz