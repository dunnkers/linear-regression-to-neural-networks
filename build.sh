build_dir='./build'

rm -rf $build_dir
mkdir $build_dir
jupyter nbconvert \
    ./src/*.ipynb \
    --to html \
    --output-dir $build_dir \
    --template ./node_modules/nbconvert-blog-template/blog
cp -avR images $build_dir