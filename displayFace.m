function displayFace(sample, width)
    sample = double(sample) / 256;
    sample = reshape(sample, width, width);
    imshow(sample);