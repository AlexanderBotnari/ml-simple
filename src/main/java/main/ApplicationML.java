package main;


import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;




// HW:
// integrate this ML example -> Spring Boot image upload
// rec:
//   *  engine init + model load -> Service constructor()
//   *  image prediction -> Service predict() 
//   *  controller -> upload -> service.predict() -> predictions -> JSON
//   *  Uppy.js (on.... response -> function(responseFromServer){ -> Document })
public class ApplicationML {
		// logger binding
	    private static final Logger logger = LoggerFactory.getLogger(ApplicationML.class);
	    private ApplicationML() {}

	    // run
	    public static void main(String[] args) throws IOException, ModelException, TranslateException {
	        DetectedObjects detection = ApplicationML.predict();
	        logger.info("{}", detection);
	    }
	    
	    // loading the image and predict
	    public static DetectedObjects predict() throws IOException, ModelException, TranslateException {
	        Path imageFile = Paths.get("images/2.jpg");
	        Image img =  ImageFactory.getInstance().fromFile(imageFile);
	        logger.info("Engine: {}", Engine.getInstance().getEngineName());
	        String backbone;
	        if ("TensorFlow".equals(Engine.getInstance().getEngineName())) {
	            backbone = "mobilenetv2";
	        } else {
	            backbone = "resnet50";
	        }
	        
	        // settings builder for the model
	        Criteria<Image, DetectedObjects> criteria =
	                Criteria.builder()
	                        .optApplication(Application.CV.OBJECT_DETECTION)
	                        .setTypes(Image.class, DetectedObjects.class)
	                        .optFilter("backbone", backbone)
	                        .optProgress(new ProgressBar())
	                        .build();

	        try (ZooModel<Image, DetectedObjects> model = criteria.loadModel()) {
	            try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
	                DetectedObjects detection = predictor.predict(img);
	                saveBoundingBoxImage(img, detection);
	                return detection;
	            }
	        }
	    }

	    private static void saveBoundingBoxImage(Image img, DetectedObjects detection)
	            throws IOException {
	        Path outputDir = Paths.get("images");
	        Files.createDirectories(outputDir);
	        logger.info("Detected objects: {}", detection.items());
	        // Make image copy with alpha channel because original image was jpg
	        Image newImage = img.duplicate(Image.Type.TYPE_INT_ARGB);
	        newImage.drawBoundingBoxes(detection);

	        Path imagePath = outputDir.resolve("detected-2.png");
	        // OpenJDK can't save jpg with alpha channel
	        newImage.save(Files.newOutputStream(imagePath), "png");
	        logger.info("Detected objects image has been saved in: {}", imagePath);
	    }

}
