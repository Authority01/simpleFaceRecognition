function displayFace(sample, width, height)
    if ~ exist('height','var') || isempty(height)
        height = width;
    end
    %sample = double(sample) / 256;
    sample = reshape(sample, width, height);
    imshow(sample);