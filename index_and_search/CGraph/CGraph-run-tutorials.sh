#!/bin/bash

CGRAPH_TUTORIAL_DIR="build/tutorial/"

if [ ! -d $CGRAPH_TUTORIAL_DIR ]; then
    echo -e "\033[31m[ERROR] Directory $CGRAPH_TUTORIAL_DIR does not exist.\033[0m"
    exit 1
fi

# shellcheck disable=SC2164
cd $CGRAPH_TUTORIAL_DIR

for file in *; do
    # 执行所有可执行文件，并且过滤掉文件夹
    if [ -f "$file" ] && [ -x "$file" ]; then
        echo "**** Running $file ****"
        "./$file"
        echo " "
    fi
done

echo -e "\033[34m congratulations, automatic run CGraph tutorial finish... \033[0m"
