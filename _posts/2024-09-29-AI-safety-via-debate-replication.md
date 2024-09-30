## Paper Replication: AI Safety via Debate - Convincing a Sparse Classifier

In this post I will go through my current progress in replicating the "convincing a sparse classifier" experiment in Section 3.1 of _AI safety via debate_ (Irving, Christiano, and Amodei 2018).
I haven't fully achieved this goal but I'm posting my current status as my project submission for the Summer 2024 installment of the _AI Safety Fundamentals_ course by BlueDot. [Here](https://github.com/Jackmastr/debate-game-replication) is a link to my github repository where I am still working on it. My current status as of the time of this post is I think I have the judge implemented correctly, and am getting similar test accuracy on random pixels in the 4 and 6 case. I've also created a version of Figure 3(a), showing the behvior of the judges errors on random input, that matches closely with the original paper. I've currently made a lot of progress on the debate agents (which use Monte Carlo Tree Search), however I'm not quite there yet and need more time to test and run them.

To be a complete replication my goal was to approximately remake most of the same figures and arrive at nearly the same results quoted in that section. To wit, that meant:
1. Figure 2 ![Figure 2](/images/AI_safety_via_debate_fig2.png)
2. Table 2 ![Table 2](/../images/AI_safety_via_debate_tab2.png)
3. Figure 3 ![Figure 3](/images/AI_safety_via_debate_fig3.png)
4. Figure 4 ![Figure 4](/images/AI_safety_via_debate_fig4.png)

There are a number of parts to the debate game, I began with the "judge" a neural network trained to classify MNIST images of the digits 0 to 9 when only given the values of 4 (6) nonzero pixels of 28x28 pixel images.

### Section 1 - Implimenting the Judge
Starting from the text of the original paper itself:
> Concretely, the judge is trained to classify MNIST from 6 (resp. 4) nonzero pixels, with the pixels chosen at random at training time. The judge receives two input feature planes: a {0, 1} mask of which pixels were revealed and the value of the revealed pixels (with zeros elsewhere). We used the architecture from the TensorFlow MNIST layers tutorial; the only difference is the input. We train the judges using Adam with a learning rate of $10^{−4}$ for 30k (resp. 50k) batches of 128 samples, reaching 59.4% (resp. 48.2%) accuracy.

Now let's break this down into actionable parts. 
1. We first need to acquire MNIST images and have a way of randomly choosing some number of random pixels from each image at training time. There should be two outputs of this preprocessing step. One is a mask of which pixels were revealed. The other is the value of the revealed pixels.
2. Next we need to reproduce the architecture used in the relevant TensorFlow MNIST layers tutorial (with a small tweak in what it takes as input). A link to this is included as a footnote in the original paper. The link is currently dead but the Wayback Machine contains an archive of the original tutorial here: [Captured May 18 2018](https://web.archive.org/web/20180516102820/https://www.tensorflow.org/tutorials/layers#building_the_cnn_mnist_classifier).
3. We train the judges using the additional information provided. The optimizer is Adam, the learning rate is $10^{−4}$, the batch size is 128 samples, the number of batches is 30k in the 6 pixel case and 50k in the 4 pixel case.
4. We assess the judges accuracy on the test set, hopefully similar to 59.4% (48.2%) in the 6 (4) pixel case.

### Section 1.1 - Preprocessing the Images
To start off I will be using the `pytorch` python package instead of `tensorflow` for the most part since I have a little bit of familiarity from doing the beginning lessons of the [Alignment Research Engineer Accelerator (ARENA) course](https://www.arena.education/). To help me (besides the usual answers you can find googling) I made use of Anthropic's Claude and OpenAI's ChatGPT either on their own or through the AI code editor Cursor. There were a number of other replications or partial replications I came across online most notably https://www.alignmentforum.org/posts/5Kv2qNfRyXXihNrx2/ai-safety-debate-and-its-applications that I consulted although I wanted to keep the implementation mine at the end of the day.

Getting that out of the way, this the way I went about loading and preprocessing the MNIST images. I defined a function that masks all but `num_pixels` nonzero pixels in the image. In some cases I see normalization applied to the pixel values to improve training. That may help here but the usual normalization I see applied is not right, the statistics that come from selecting a few nonzero pixels aren't the same as those of the full images.
```python
from torchvision import datasets, transforms
# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    MaskAllButNPixels(num_pixels)
])

# Load MNIST dataset
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
```
In defining the transforming you need to be careful. Remember we want the output of the preprossing step to be formated like `[mask, image]` and the random pixels should be chosen without replacement and **only** be selected from the **nonzero** pixels in the image. With that done we can move on to the second step.
### Section 1.2 - Implementing the Neural Network
From the TensorFlow Tutorial:
>Convolutional Layer #1: Applies 32 5x5 filters (extracting 5x5-pixel subregions), with ReLU activation function
Pooling Layer #1: Performs max pooling with a 2x2 filter and stride of 2 (which specifies that pooled regions do not overlap)
Convolutional Layer #2: Applies 64 5x5 filters, with ReLU activation function
Pooling Layer #2: Again, performs max pooling with a 2x2 filter and stride of 2
Dense Layer #1: 1,024 neurons, with dropout regularization rate of 0.4 (probability of 0.4 that any given element will be dropped during training)
Dense Layer #2 (Logits Layer): 10 neurons, one for each digit target class (0–9).

My implementation for reference:
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        # 1. Convolutional Layer #1: Applies 32 5x5 filters (extracting 5x5-pixel subregions), with ReLU activation function
        x = torch.relu(self.conv1(x))
        # 2. Pooling Layer #1: Performs max pooling with a 2x2 filter and stride of 2 (which specifies that pooled regions do not overlap)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        # 3. Convolutional Layer #2: Applies 64 5x5 filters, with ReLU activation function
        x = torch.relu(self.conv2(x))
        # 4. Pooling Layer #2: Again, performs max pooling with a 2x2 filter and stride of 2
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 64 * 7 * 7)
        # 5. Dense Layer #1: 1,024 neurons, with dropout regularization rate of 0.4 (probability of 0.4 that any given element will be dropped during training)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        # 6. Dense Layer #2 (Logits Layer): 10 neurons, one for each digit target class (0–9).
        x = self.fc2(x)
        return x
```
### Section 1.3 - Training the Model
The MNIST training data set appears to have 60,000 entries. With a batch size of 128 this means once pass through the entire set is 60000/128 = 468.75 $\approx$ 469 batches/epoch. To reach at least 30k (60k) batches total means doing 30k/469 $\approx$ 64 batches or 50k/469 $\approx$ 107 batches respectively. Based on my results I think this is how the number of batches is suppoed to be interpreted. Knowing this it's straightforward to train with the given hyperparameters.
### Section 1.4 - Training Results
OK. So now the (first) moment of truth. Did these model give roughly the same accuracy as the original paper on a test dataset preprocessed the same way as the training dataset? More or less yes! For 6 (4) pixels my judge achieved 60.0% (48.5%) vs the paper's 59.4% (48.2%)! For more fine grained results I implemented a version of Figure 3(a) in the paper (for the 6 pixel case). Here are the two side-by-side:
![my version](/images/error_matrix6.png) ![paper version](/images/AI_safety_via_debate_fig3a.png)

It's not in the original paper but here is the version for 4 pixels: ![my version](/images/error_matrix4.png)
In the text it's refered to as a "confusion matrix" although I'm not sure that always means the same thing to different authors. To be more explicit: the percentage in each square is the percentage of **all** images in the test dataset with that true label and that predicted label. If you summed the value in every square (including the hidden numbers along the diagonal) it adds up to 100%.
