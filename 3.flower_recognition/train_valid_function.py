import torch
from tensorboardX import SummaryWriter


def adjust_learning_rate(optimizer, gamma, global_step):
    lr = 1e-3 * (gamma ** global_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validation(model, valid_loader, device):
    cnt = 0  # validation中所有图片的数量
    total_acc = 0  # 所有图片的accuracy的和
    model.to(device)

    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            cnt += images.shape[0]
            images, labels = images.to(device), labels.to(device)
            model_output = model(images)
            model_label = torch.max(model_output, dim=1)[1]

            temp_acc = torch.eq(model_label, labels).sum().item()
            total_acc += temp_acc

    # print输出的是在每一张图片上的accuracy
    print('accuracy on validation set: %.6f' % (total_acc / cnt))


def train(model, train_loader, valid_loader, epochs, optimizer, criterion, device, valid_interval, save_interval):
    model.to(device)
    train_step = 0  # iter的个数
    total_loss = 0  # save_interval个iter的loss总和
    total_acc = 0  # save_interval个iter的accuracy总和
    lr_adjust_step = 0
    writer = SummaryWriter(log_dir='./save_model/summary')

    for epoch in range(epochs):

        if epoch + 1 == 6:
            lr_adjust_step += 1

        for index, (images, labels) in enumerate(train_loader):
            train_step += 1
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            model_output = model(images)
            loss = criterion(model_output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # 计算accuracy
            model_label = torch.max(model_output, dim=1)[1]
            temp_acc = torch.eq(model_label, labels).sum().item()
            total_acc += temp_acc

            if train_step % valid_interval == 0:
                # 在validation上测试一次
                validation(model, valid_loader, device)

            if train_step % save_interval == 0:
                # 在writer中保存accuracy和loss的值
                temp_acc = total_acc / (save_interval * labels.shape[0])
                temp_loss = total_loss / (save_interval * labels.shape[0])
                print('EPOCHS: %d/%d || train_loss: %.6f || train_acc: %.6f' % (epoch+1, epochs, temp_loss, temp_acc))

                writer.add_scalar('train_accuracy', temp_acc, train_step)
                writer.add_scalar('train_loss', temp_loss, train_step)
                total_loss = 0
                total_acc = 0

        # 每一个epoch保存一次模型
        torch.save(model.state_dict(), './save_model/VGG16_flower_' + str(epoch + 1) + '.pth')

    writer.close()



