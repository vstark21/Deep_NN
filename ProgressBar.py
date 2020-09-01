import sys

# Progress Bar class to output the progress
class ProgressBar:


    def __init__(self, total_count):
        self.total_count = total_count



    def update(self, present_count):

        percent = int((present_count / self.total_count) * 25) + 1
        sys.stdout.write('\r')
        sys.stdout.write("Epoch : [ " + "> " * percent + "= " * (25 - percent) + "] " + str(present_count) + " / " + str(self.total_count))



    def update_with_loss(self, present_count, loss_acc):

        loss, accuracy = loss_acc
        percent = int((present_count / self.total_count) * 25) + 1
        sys.stdout.write('\r')
        sys.stdout.write("Epoch : [ " + "> " * percent + "= " * (25 - percent) + "] " + str(present_count) + " / " + str(self.total_count) + " Loss: " + str(loss) + " Accuracy: " + str(accuracy))



    def end(self):

        print()
