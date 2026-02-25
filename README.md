<h1 align="center">AutoEncoder Image Compression</h1>

<hr>

<h2>Project Overview</h2>

<p><strong>Language Used:</strong> Python</p>
<p><strong>Libraries Used:</strong> PyTorch (torch), TorchVision (torchvision), Matplotlib (matplotlib)</p>
<p><strong>Dataset Used:</strong> Fashion-MNIST (contains 60,000 grayscale images)</p>

<hr>

<h2>Workflow</h2>

<h3>1️⃣ Defining the Model Class</h3>

<p>
The <strong>SimpleAutoencoder</strong> class inherits from <code>nn.Module</code>, the base class for all neural network implementations in PyTorch.
It provides built-in methods such as <code>train()</code> and <code>eval()</code>.
</p>

<p>
The constructor defines:
</p>

<ul>
  <li><strong>input_size</strong>: 28 × 28 image flattened into a vector.</li>
  <li><strong>hidden_size</strong>: Dimension of compressed representation (bottleneck).</li>
</ul>

<p>
The <strong>Encoder</strong> performs linear transformation from high-dimensional input to lower dimension.
The <strong>Decoder</strong> reconstructs the image back to original dimension.
</p>

<p>
The <strong>ReLU activation function</strong> applies <code>max(0, x)</code> to introduce sparsity and improve feature learning efficiency.
</p>

<p>
The <code>forward()</code> function:
</p>

<ul>
  <li>Flattens 2D images into 1D vectors using <code>view()</code></li>
  <li>Passes data through encoder → activation → decoder</li>
  <li>Returns reconstructed image</li>
</ul>

<hr>

<h3>2️⃣ Loading Data for Training & Testing</h3>

<ul>
  <li>Separate datasets used for training and testing (ensures generalization).</li>
  <li><code>ToTensor()</code> normalizes pixel values to range [0,1].</li>
  <li><code>DataLoader</code> enables batch processing.</li>
  <li>Training data is shuffled to prevent order-based learning and overfitting.</li>
</ul>

<hr>

<h3>3️⃣ Loss Function & Optimization</h3>

<p><strong>Loss Function:</strong> Mean Squared Error (MSE)</p>
<p>
MSE calculates the average squared difference between reconstructed and original images.
</p>

<p><strong>Optimizer:</strong> Adam</p>
<p>
Adam adjusts weights based on computed gradients.
</p>

<ul>
  <li><strong>Learning Rate (lr):</strong> Controls speed of learning.</li>
  <li><strong>Epochs:</strong> One full pass through dataset.</li>
</ul>

<p>
Too many epochs → Overfitting <br>
Too few epochs → Underfitting
</p>

<hr>

<h3>4️⃣ Training Loop</h3>

<ul>
  <li><code>model.train()</code> enables training mode.</li>
  <li><code>zero_grad()</code> resets accumulated gradients.</li>
  <li>Forward pass → Loss calculation → Backpropagation.</li>
  <li><code>optimizer.step()</code> updates model weights.</li>
  <li>Loss printed epoch-wise.</li>
</ul>

<p>
Since this is <strong>unsupervised learning</strong>, labels are not used.
</p>

<hr>

<h3>5️⃣ Testing Loop</h3>

<ul>
  <li><code>model.eval()</code> disables training-specific layers.</li>
  <li><code>torch.no_grad()</code> improves efficiency by disabling gradient computation.</li>
  <li>Test loss calculated on entire test dataset.</li>
</ul>

<hr>

<h3>6️⃣ Displaying Reconstructed Images</h3>

<ul>
  <li>First test image selected.</li>
  <li><code>unsqueeze()</code> adds batch dimension.</li>
  <li><code>view()</code> reshapes tensors back to 28×28.</li>
  <li><code>matplotlib.pyplot</code> used for visualization.</li>
</ul>

<p>
<strong>Matplotlib functions used:</strong>
</p>

<ul>
  <li><code>figure()</code> – Sets display size</li>
  <li><code>subplot()</code> – Arranges image positions</li>
  <li><code>imshow()</code> – Displays image</li>
  <li><code>axis("off")</code> – Removes axes</li>
</ul>

<hr>

<h2>Applications</h2>

<ul>
  <li>Image / Audio Compression</li>
  <li>Pattern Recognition</li>
  <li>Database Storage Optimization</li>
  <li>Anomaly Detection</li>
  <li>Image Restoration</li>
</ul>

<hr>
