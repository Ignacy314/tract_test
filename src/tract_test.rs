use tract_onnx::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = tract_onnx::onnx()
        // load the model
        .model_for_path("r50.onnx")?
        // specify input type and shape
        .with_input_fact(0, f32::fact([1, 224, 224, 3]).into())?
        // optimize the model
        .into_optimized()?
        // make the model runnable and fix its inputs and outputs
        .into_runnable()?;

    // open image, resize it and make a Tensor out of it
    let image = image::open("images/0/s164_3.png").unwrap().to_rgb8();
    let resized =
        image::imageops::resize(&image, 224, 224, ::image::imageops::FilterType::Triangle);
    #[allow(clippy::cast_possible_truncation)]
    let image: Tensor = tract_ndarray::Array4::from_shape_fn((1, 224, 224, 3), |(_, y, x, c)| {
        f32::from(resized[(x as _, y as _)][c]) / 255.0
    })
    .into();

    // run the model on the input
    let result = model.run(tvec!(image.into()))?;

    // find and display the max value with its index
    //let best = result[0]
    //    .to_array_view::<f32>()?
    //    .iter()
    //    .copied()
    //    .zip(1..)
    //    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    println!("result: {:?}", result[0].to_array_view::<f32>());
    Ok(())
}
