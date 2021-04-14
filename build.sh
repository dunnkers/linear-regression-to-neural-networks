build_dir='./build'

rm -rf $build_dir
mkdir $build_dir
cp -avR src/* $build_dir
cp -avR public/* $build_dir
jupyter nbconvert \
    ./build/*.ipynb \
    --to html \
    --output-dir $build_dir \
    --template ./node_modules/nbconvert-blog-template/blog
rm -rf $build_dir/*.ipynb