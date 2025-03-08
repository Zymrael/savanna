# Unicode checkmark
checkmark="✔" #"\xE2\x9C\x94"
# Unicode crossmark
crossmark="✘" #"\xE2\x9C\x98"

check_substring() {
  local string="$1"
  local substring="$2"

  if [[ "$string" != *"$substring"* ]]; then
    echo "${crossmark} '$substring' not found in '$string'. Exiting."
    exit 1
  else
    echo "${checkmark} '$substring' found in '$string'"
  fi
}

check_directory_does_not_exist() {
  local dir_path="$1"

  if [ -d "$dir_path" ]; then
  # insert unicode checkmark
    echo "${crossmark} '$dir_path' already exists. Exiting!"
    exit 1  
  else
    echo "${checkmark} '$dir_path' does not exist. Proceeding."
  fi
}

check_directory_exists() {
  local dir_path="$1"

  if [ -d "$dir_path" ]; then
    echo "${checkmark} '$dir_path' exists. Proceeding!"
  else
    printf "${crossmark} '$dir_path' does not exist. Exiting!"
    exit 1
  fi
}