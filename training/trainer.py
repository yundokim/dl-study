from core import Variable, to_onehot, SoftmaxCrossEntropy

def train(model, optimizer, x_train, y_train, batch_size=100):
    for i in range(0, len(x_train), batch_size):
        xb = Variable(x_train[i:i + batch_size])
        labels = y_train[i:i + batch_size]
        yb = Variable(to_onehot(labels))

        optimizer.zero_grad()
        out = model(xb)
        loss = SoftmaxCrossEntropy()(out, yb)
        loss.backward()
        optimizer.step()