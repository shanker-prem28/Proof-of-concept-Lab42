import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

matplotlib_is_available = True
try:
  from matplotlib import pyplot as plt
except ImportError:
  print("Will skip plotting; matplotlib is not available.")
  matplotlib_is_available = False

# Data params
data_mean = 4
data_stddev = 1.25

(name, preprocess, d_input_func) = ("Only 4 moments", lambda data: get_moments(data), lambda x: 4)

print("Using data [%s]" % (name))

# ##### DATA######################################################################

def get_distribution_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))  

def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n) 

# ##### MODELS #################################################################

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, x):
        x = self.map1(x)
        x = self.f(x)
        x = self.map2(x)
        x = self.f(x)
        x = self.map3(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, x):
        x = self.f(self.map1(x))
        x = self.f(self.map2(x))
        return self.f(self.map3(x))

def extract(v):
    return v.data.storage().tolist()

def stats(d):
    return [np.mean(d), np.std(d)]

def get_moments(d):
    
    mean = torch.mean(d)
    diffs = d - mean
    var = torch.mean(torch.pow(diffs, 2.0))
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0))
    kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0  # excess kurtosis, should be 0 for Gaussian
    final = torch.cat((mean.reshape(1,), std.reshape(1,), skews.reshape(1,), kurtoses.reshape(1,)))
    return final

def decorate_with_diffs(data, exponent, remove_raw_data=False):
    mean = torch.mean(data.data, 1, keepdim=True)
    mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
    diffs = torch.pow(data - Variable(mean_broadcast), exponent)
    if remove_raw_data:
        return torch.cat([diffs], 1)
    else:
        return torch.cat([data, diffs], 1)

def train():
   
    generator_input_size = 1      # Random noise dimension coming into generator
    generator_hidden_size = 5     # Generator complexity
    generator_output_size = 1     # Size of generated output vector
    discriminator_input_size = 500    # Minibatch size - cardinality of distributions
    discriminator_hidden_size = 10    # Discriminator complexity
    discriminator_output_size = 1     # Single dimension for 'real' vs. 'fake' classification
    minibatch_size = discriminator_input_size

    discriminator_learning_rate = 1e-3
    generator_learning_rate = 1e-3
    sgd_momentum = 0.9

    num_epochs = 5000
    print_interval = 100
    discriminator_steps = 20
    generator_steps = 20

    dfe, dre, ge = 0, 0, 0
    discriminator_real_data, d_fake_data, g_fake_data = None, None, None

    discriminator_af = torch.sigmoid #activation function for the discriminator.
    
    generator_af= torch.tanh # activation function for the generator.

    discriminator_sampler = get_distribution_sampler(data_mean, data_stddev)
    gi_sampler = get_generator_input_sampler()
    G = Generator(input_size=generator_input_size,
                  hidden_size=generator_hidden_size,
                  output_size=generator_output_size,
                  f=generator_af)
    D = Discriminator(input_size=d_input_func(discriminator_input_size),
                      hidden_size=discriminator_hidden_size,
                      output_size=discriminator_output_size,
                      f=discriminator_af)
    crit = nn.BCELoss()  # Binary cross entropy, Loss function.
    discriminator_optimizer = optim.SGD(D.parameters(), lr=discriminator_learning_rate, momentum=sgd_momentum)
    generator_optimizer = optim.SGD(G.parameters(), lr=generator_learning_rate, momentum=sgd_momentum)

    for epoch in range(num_epochs):
        for d_index in range(discriminator_steps):
            #Train Discriminator on real+fake
            D.zero_grad()

            #Train Discriminator on real
            discriminator_real_data = Variable(discriminator_sampler(discriminator_input_size))
            d_real_decision = D(preprocess(discriminator_real_data))
            d_real_error = crit(d_real_decision, Variable(torch.ones([1,1])))  # ones = true
            d_real_error.backward() # compute/store gradients, but don't change params

            #Train Discriminator on fake
            d_gen_input = Variable(gi_sampler(minibatch_size, generator_input_size))
            d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
            d_fake_decision = D(preprocess(d_fake_data.t()))
            d_fake_error = crit(d_fake_decision, Variable(torch.zeros([1,1])))  # zeros = fake
            d_fake_error.backward()
            discriminator_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

            dre, dfe = extract(d_real_error)[0], extract(d_fake_error)[0]

        for g_index in range(generator_steps):
            #Train Generator on Discriminator's response (but DO NOT train D on these labels)
            G.zero_grad()

            gen_input = Variable(gi_sampler(minibatch_size, generator_input_size))
            g_fake_data = G(gen_input)
            dg_fake_decision = D(preprocess(g_fake_data.t()))
            g_error = crit(dg_fake_decision, Variable(torch.ones([1,1])))  #Train Generator to pretend it's genuine

            g_error.backward()
            generator_optimizer.step()  #Only optimizes Generator's parameters
            ge = extract(g_error)[0]

        if epoch % print_interval == 0:
            print("Epoch %s: D (%s real_err, %s fake_err) G (%s err); Real Dist (%s),  Fake Dist (%s) " %
                  (epoch, dre, dfe, ge, stats(extract(discriminator_real_data)), stats(extract(d_fake_data))))

    if matplotlib_is_available: #Proof_of_Concept #lab42
        print("Plotting the generated distribution...")
        values = extract(g_fake_data)
        print(" Values: %s" % (str(values)))
        plt.hist(values, bins=50)
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.title('Histogram')
        plt.grid(True)
        plt.show()


train()