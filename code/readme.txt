****Testing Instructions****

1. Keep whatever video file you want to test in the "test" folder.
2. Run final.py

****Files included in CODE folder****
1. cnn_model.h5 -> Pre-trained CNN model
2. final.py -> Testing file
3. handshape_feature_extractor -> Pre-processes input given to CNN
4. output_labels_alphabet.txt -> list of labels for comparison
5. google_drive_links.txt -> Alphabet and Word data created by every member of this group

****Files included in DEMO folder****
1. Demo.mp4 -> Contains our demo video for our project detailing every step we executed.

****Files included in REPORT folder****
1. Mobile_Computing_ASL_Project_Report.pdf -> Our final project report

****Supporting Scripts****
1. video_to_frames.py -> converts video file to frames
2. frames_to_keypoints.js -> Javascript file to create keypoints.json file from posenet
3. json_to_coordinates.py -> Extract wrist coordinates from the json file
4. coordinates_to_cropped.py -> creates a cropped image from original frames
5. images_to_labels.py -> Calculates labels for each input frame
6. word_segmentation.py -> Segments labels based on each letter and outputs final merged word
