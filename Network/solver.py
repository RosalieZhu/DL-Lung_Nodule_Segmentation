import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
import csv

# Additional imports (R&R)
#from lr_scheduler import ReduceLROnPlateau
from matplotlib import pyplot as plt

class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader, whole_slice_prediction_loader):

		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader
		self.whole_slice_prediction_loader = whole_slice_prediction_loader

		# Models
		self.unet = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		#self.criterion = torch.nn.BCELoss()
		self.augmentation_prob = config.augmentation_prob
		self.inverse_ratio = config.inverse_ratio

		# Hyper-parameters
		self.initial_lr = config.lr
		self.current_lr = config.lr
        
		self.optimizer_choice = config.optimizer_choice
		if config.optimizer_choice == 'Adam':
			self.beta1 = config.beta1
			self.beta2 = config.beta2
		elif config.optimizer_choice == 'SGD':
			self.momentum = config.momentum
		else:
			print('No such optimizer available')

		# Training settings
		self.num_epochs = config.num_epochs
		#self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size
		self.PPorLS = config.PPorLS

		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step
		self.batch_val_num = config.val_freq_batch

		# Path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.result_img_path = config.result_img_path
		self.mode = config.mode

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_type = config.model_type
		self.t = config.t
		self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type =='U_Net':
			self.unet = U_Net(img_ch=1,output_ch=1)
		elif self.model_type =='R2U_Net':
			self.unet = R2U_Net(img_ch=1,output_ch=1,t=self.t)
		elif self.model_type =='AttU_Net':
			self.unet = AttU_Net(img_ch=1,output_ch=1)
		elif self.model_type == 'R2AttU_Net':
			self.unet = R2AttU_Net(img_ch=1,output_ch=1,t=self.t)
			
		if self.optimizer_choice == 'Adam':
			self.optimizer = optim.Adam(list(self.unet.parameters()), self.initial_lr, [self.beta1, self.beta2])
		elif self.optimizer_choice == 'SGD':
			self.optimizer = optim.SGD(list(self.unet.parameters()), self.initial_lr, self.momentum)
		else:
			pass

		self.unet.to(self.device)

		#self.print_network(self.unet, self.model_type)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))


	def dice_coeff_loss(self, y_pred, y_true):
		smooth =1
		y_true_flat = y_true.view(y_true.size(0), -1)
		y_pred_flat = y_pred.view(y_pred.size(0), -1)
		intersection = (y_true_flat * y_pred_flat).sum()
        
		return - (2. * intersection + smooth) / ((y_true_flat).sum() + (y_pred_flat).sum() + smooth)


	def RR_dice_coeff_loss(self, y_pred, y_true):
		smooth = 1e-6
		y_true_flat = y_true.view(y_true.size(0), -1)
		y_pred_flat = y_pred.view(y_pred.size(0), -1)
		intersection = (y_true_flat * y_pred_flat).sum()
        
		inverse_y_true_flat = 1 - y_true_flat
		inverse_y_pred_flat = 1 - y_pred_flat
		inverse_intersection = (inverse_y_true_flat * inverse_y_pred_flat).sum()
		return - (2. * intersection + smooth) / ((y_true_flat).sum() + (y_pred_flat).sum() + smooth) - (2. * inverse_intersection + smooth) / ((inverse_y_true_flat).sum() + (inverse_y_pred_flat).sum() + smooth)

	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	# Redefine the 'update_lr' function (R&R)
	def update_lr(self, new_lr):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = new_lr
            
            
	def run_batch_validation(self, epoch, batch_train):
		self.unet.train(False)
		self.unet.eval()

        
		acc = 0.	# Accuracy
		SE = 0.		# Sensitivity (Recall)
		SP = 0.		# Specificity
		PC = 0. 	# Precision
		F1 = 0.		# F1 Score
		JS = 0.		# Jaccard Similarity
		DC = 0.		# Dice Coefficient
		DC_RR = 0
		length=0
        
		validation_batch_loss = 0
		for batch, (images, GT) in enumerate(self.valid_loader):

			images = images.to(self.device)
			GT = GT.to(self.device)
			# Reshape the images and GT to 4-dimensional so that they can get fed to the conv2d layer.
			images = images.reshape(self.batch_size, self.img_ch, np.shape(images)[1], np.shape(images)[2])
			GT = GT.reshape(self.batch_size, self.img_ch, np.shape(GT)[1], np.shape(GT)[2])

			#SR = F.sigmoid(self.unet(images))
			SR = torch.sigmoid(self.unet(images))
			acc += get_accuracy(SR,GT)
			SE += get_sensitivity(SR,GT)
			SP += get_specificity(SR,GT)
			PC += get_precision(SR,GT)
			F1 += get_F1(SR,GT)
			JS += get_JS(SR,GT)
			DC_RR += get_DC_RR(SR,GT, inverse_ratio = self.inverse_ratio)
			DC += get_DC(SR, GT)
						
			length += images.size(0)

			# Compute the validation loss.
			SR = self.unet(images)
			SR_probs = torch.sigmoid(SR)
			SR_flat = SR_probs.view(SR_probs.size(0),-1)
			GT_flat = GT.view(GT.size(0),-1)
			# use the dice coefficient loss instead of the BCE loss. (R&R)
			validation_loss = self.dice_coeff_loss(SR_flat, GT_flat)

			validation_batch_loss += validation_loss.item()

		acc = acc/length
		SE = SE/length
		SP = SP/length
		PC = PC/length
		F1 = F1/length
		JS = JS/length
		DC = DC/length
		DC_RR = DC_RR/length
		unet_score = DC_RR

		print('current batch: {}'.format(batch_train))
		print('Current learning rate: {}'.format(self.current_lr))
        
		print('Current Batch [%d] \n[Validation] Validation Loss: %.4f, Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, DC_RR: %.4f' % (batch_train + 1, validation_batch_loss, acc, SE, SP, PC, F1, JS, DC, DC_RR))

		# Append validation loss to train loss history (R&R)
				
	
		f = open(os.path.join(self.result_path, 'model_validation_batch_history.csv'), 'a', encoding = 'utf-8', newline= '')
		wr = csv.writer(f)
		wr.writerow(['Validation', 'Epoch [%d/%d]' % (epoch + 1, self.num_epochs), 'Batch [%d]' % (batch_train + 1), 'Validation loss: %.4f' % validation_batch_loss,'Accuracy: %.4f' % acc, 'Sensitivity: %.4f' % SE, 'Specificity: %.4f' % SP, 'Precision: %.4f'% PC, 'F1 Score: %.4f' % F1, 'Jaccard Similarity: %.4f' % JS, 'Dice Coefficient: %.4f' % DC, 'RR_DC: %.4f' % DC_RR])
        
		self.unet.train(True)
        
		return(validation_batch_loss, unet_score)

	# Define adaptive learning rate handler (R&R)
	def adaptive_lr_handler(self, cooldown, min_lr, current_epoch, previous_update_epoch, plateau_ratio, adjustment_ratio, loss_history):
		if current_epoch > 1:
			if current_epoch - previous_update_epoch > cooldown:
				if (loss_history[-1] > loss_history[-2]) or (abs((loss_history[-2] - loss_history[-1])/loss_history[-2]) < plateau_ratio):
					if self.current_lr > min_lr:
						self.current_lr = adjustment_ratio * self.current_lr
						self.update_lr(self.current_lr)
						print('Validation loss stop decreasing. Adjust the learning rate to {}.'.format(self.current_lr))
						return current_epoch
    
    

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.unet.zero_grad()

	def tensor2img(self,x):
		img = (x[:,0,:,:]>x[:,1,:,:]).float()
		img = img*255
		return img

	def train(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#

		unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%.4f-%s-%s-%.4f.pkl' %(self.model_type,self.num_epochs,self.initial_lr,self.augmentation_prob, self.PPorLS, self.optimizer_choice, self.inverse_ratio))
		print('The U-Net path is {}'.format(unet_path))
		# U-Net Train
		# Train loss history (R&R)
		train_loss_history = []
		train_batch_loss_history = []
		# Validation loss history (R&R)
		validation_loss_history = []
		val_batch_loss_history = []
		stop_training = False

		if os.path.isfile(unet_path):
			# Load the pretrained Encoder
			self.unet.load_state_dict(torch.load(unet_path))
			print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
		else:
			# Train for Encoder
			best_unet_score = 0.
			print('Start training. The initial learning rate is: {}'.format(self.initial_lr))

			for epoch in range(self.num_epochs):
				self.unet.train(True)
				train_epoch_loss = 0; validation_epoch_loss = 0

				if stop_training == True:
					break
				acc = 0.	# Accuracy
				SE = 0.		# Sensitivity (Recall)
				SP = 0.		# Specificity
				PC = 0. 	# Precision
				F1 = 0.		# F1 Score
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				DC_RR = 0
				length = 0

				for batch, (images, GT) in enumerate(self.train_loader):
					# GT : Ground Truth
					images = images.to(self.device)
					GT = GT.to(self.device)
					# Reshape the images and GT to 4-dimensional so that they can get fed to the conv2d layer. (R&R)
					images = images.reshape(self.batch_size, self.img_ch, np.shape(images)[1], np.shape(images)[2])
					GT = GT.reshape(self.batch_size, self.img_ch, np.shape(GT)[1], np.shape(GT)[2])

					# SR : Segmentation Result
					SR = self.unet(images)
					SR_probs = torch.sigmoid(SR)
					SR_flat = SR_probs.view(SR_probs.size(0), -1)
					
					GT_flat = GT.view(GT.size(0), -1)
					# Use dice coefficient loss instead of the BCE loss. (R&R)
					train_loss = self.dice_coeff_loss(SR_flat, GT_flat)

					train_epoch_loss += train_loss.item()

					# Backprop + optimize
					self.reset_grad()
					train_loss.backward()
					self.optimizer.step()
					
					acc += get_accuracy(SR,GT)
					SE += get_sensitivity(SR,GT)
					SP += get_specificity(SR,GT)
					PC += get_precision(SR,GT)
					F1 += get_F1(SR,GT)
					JS += get_JS(SR,GT)
					DC_RR += get_DC_RR(SR,GT, inverse_ratio = self.inverse_ratio)
					DC += get_DC(SR, GT)
					length += images.size(0)
                    
					if epoch == 0:
						val_frequency = self.batch_val_num[0]
					else:
						val_frequency = self.batch_val_num[1]
                    
					if batch % val_frequency == 0:
                        # update learning rate and record the validation loss history
						validation_batch_loss, unet_score = self.run_batch_validation(epoch, batch)
						val_batch_loss_history.append(validation_batch_loss)
						train_batch_loss_history.append(train_epoch_loss)
                        
						if unet_score > best_unet_score:
							best_unet_score = unet_score
							best_epoch = epoch
							best_unet = self.unet.state_dict()
							print('Best %s model score : %.4f'%(self.model_type,best_unet_score))
							torch.save(best_unet,unet_path)

                        # update learning rate
						batch_id = len(val_batch_loss_history)
						try:
							previous_batch_id = self.adaptive_lr_handler(3, 0.01*self.initial_lr, batch_id, previous_batch_id, 0.001, 0.5, val_batch_loss_history)
						except:
							previous_batch_id = self.adaptive_lr_handler(3, 0.01*self.initial_lr, batch_id, 0, 0.001, 0.5, val_batch_loss_history)
                            
						if ((batch_id - 4) % 10 == 0) and (batch_id > 8) or unet_score < 0.2 * best_unet_score:
							if (np.median(val_batch_loss_history[-10:-5]) >= np.median(val_batch_loss_history[-5:])):
								print('Validation loss stop decreasing. Stop training.')
								stop_training = True								
								break
                
				if stop_training == True:
					break
                
				acc = acc/length
				SE = SE/length
				SP = SP/length
				PC = PC/length
				F1 = F1/length
				JS = JS/length
				DC = DC/length
				DC_RR = DC_RR/length


				# Print the log info
				print('Epoch [%d/%d] \n[Training] Train Loss: %.4f, Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, DC_RR: %.4f' % (\
                        epoch + 1, self.num_epochs, train_epoch_loss,\
					  acc, SE, SP, PC, F1, JS, DC, DC_RR))

				# Append train loss to train loss history (R&R)
				train_loss_history.append(train_epoch_loss)

                
				f = open(os.path.join(self.result_path, 'train_and_validation_history.csv'), 'a', \
                         encoding = 'utf-8', newline= '')
				wr = csv.writer(f)
				wr.writerow(['Training', 'Epoch [%d/%d]' % (epoch + 1, self.num_epochs), \
                             'Train loss: %.4f' % train_epoch_loss,\
                            'Accuracy: %.4f' % acc, 'Sensitivity: %.4f' % SE, 'Specificity: %.4f' % SP, 'Precision: %.4f'% PC, \
                            'F1 Score: %.4f' % F1, 'Jaccard Similarity: %.4f' % JS, 'Dice Coefficient: %.4f' % DC, 'RR_DC: %.4f' % DC_RR])
				f.close()

				#===================================== Validation ====================================#
				self.unet.train(False)
				self.unet.eval()

				acc = 0.	# Accuracy
				SE = 0.		# Sensitivity (Recall)
				SP = 0.		# Specificity
				PC = 0. 	# Precision
				F1 = 0.		# F1 Score
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				DC_RR = 0
				length=0

				for batch, (images, GT) in enumerate(self.valid_loader):

					images = images.to(self.device)
					GT = GT.to(self.device)
					# Reshape the images and GT to 4-dimensional so that they can get fed to the conv2d layer.
					images = images.reshape(self.batch_size, self.img_ch, np.shape(images)[1], np.shape(images)[2])
					GT = GT.reshape(self.batch_size, self.img_ch, np.shape(GT)[1], np.shape(GT)[2])

					#SR = F.sigmoid(self.unet(images))
					SR = torch.sigmoid(self.unet(images))
					acc += get_accuracy(SR,GT)
					SE += get_sensitivity(SR,GT)
					SP += get_specificity(SR,GT)
					PC += get_precision(SR,GT)
					F1 += get_F1(SR,GT)
					JS += get_JS(SR,GT)
					DC_RR += get_DC_RR(SR,GT, inverse_ratio = self.inverse_ratio)
					DC += get_DC(SR, GT)
						
					length += images.size(0)

					# Compute the validation loss.
					SR = self.unet(images)
					SR_probs = torch.sigmoid(SR)
					SR_flat = SR_probs.view(SR_probs.size(0),-1)
					GT_flat = GT.view(GT.size(0),-1)
					# use the dice coefficient loss instead of the BCE loss. (R&R)
					validation_loss = self.dice_coeff_loss(SR_flat, GT_flat)
                    

					validation_epoch_loss += validation_loss.item()

				acc = acc/length
				SE = SE/length
				SP = SP/length
				PC = PC/length
				F1 = F1/length
				JS = JS/length
				DC = DC/length
				DC_RR = DC_RR/length
				unet_score = DC_RR
				print('Current learning rate: {}'.format(self.current_lr))

				print('Epoch [%d/%d] \n[Validation] Validation Loss: %.4f, Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, DC_RR: %.4f' % (epoch + 1, self.num_epochs, validation_epoch_loss,acc, SE, SP, PC, F1, JS, DC, DC_RR))

				# Append validation loss to train loss history (R&R)
				validation_loss_history.append(validation_epoch_loss)
				
				'''
				torchvision.utils.save_image(images.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_image.png'%(self.model_type,epoch+1)))
				torchvision.utils.save_image(SR.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
				torchvision.utils.save_image(GT.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_GT.png'%(self.model_type,epoch+1)))
				'''
				
				f = open(os.path.join(self.result_path, 'train_and_validation_history.csv'), 'a', \
                         encoding = 'utf-8', newline= '')
				wr = csv.writer(f)
				wr.writerow(['Validation', 'Epoch [%d/%d]' % (epoch + 1, self.num_epochs), \
                             'Validation loss: %.4f' % validation_epoch_loss,\
                            'Accuracy: %.4f' % acc, 'Sensitivity: %.4f' % SE, 'Specificity: %.4f' % SP, 'Precision: %.4f'% PC, \
                            'F1 Score: %.4f' % F1, 'Jaccard Similarity: %.4f' % JS, 'Dice Coefficient: %.4f' % DC, 'RR_DC: %.4f' % DC_RR])
				f.close()

				# Save Best U-Net model
				if unet_score > best_unet_score:
					best_unet_score = unet_score
					best_epoch = epoch
					best_unet = self.unet.state_dict()
					print('Best %s model score : %.4f'%(self.model_type,best_unet_score))
					torch.save(best_unet,unet_path)

				# Early stop (R&R)
				#if (epoch > 8) and ((epoch - 4) % 5 == 0):
				#	if (np.median(validation_loss_history[-10:-5]) >= np.median(validation_loss_history[-5:])):
				#		print('Validation loss stop decreasing. Stop training.')
				#		break

				if (len(validation_loss_history) > 1):
					if (validation_loss_history[-2] >= validation_loss_history[-1]):
						print('Validation loss stop decreasing. Stop training.')
						break
       
		del self.unet
		try:
			del best_unet
		except:
			print('Cannot delete the variable "best_unet": variable does not exist.')
        
		return train_loss_history, validation_loss_history, val_batch_loss_history, train_batch_loss_history


	def test(self):

		"""Test encoder, generator and discriminator."""
		#======================================= Test ====================================#
		#=================================================================================#
		unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%.4f-%s-%s-%.4f.pkl' %(self.model_type,self.num_epochs,self.initial_lr,self.augmentation_prob, self.PPorLS, self.optimizer_choice, self.inverse_ratio))
		self.build_model()
		self.unet.load_state_dict(torch.load(unet_path))
		
		self.unet.train(False)
		self.unet.eval()
		acc = 0.	# Accuracy
		SE = 0.		# Sensitivity (Recall)
		SP = 0.		# Specificity
		PC = 0. 	# Precision
		F1 = 0.		# F1 Score
		JS = 0.		# Jaccard Similarity
		DC = 0.		# Dice Coefficient
		DC_RR = 0
		length=0
		for i, (images, GT) in enumerate(self.test_loader):
			images = images.to(self.device)
			GT = GT.to(self.device)
			# Reshape the images and GT to 4-dimensional so that they can get fed to the conv2d layer.
			images = images.reshape(self.batch_size, self.img_ch, np.shape(images)[1], np.shape(images)[2])
			GT = GT.reshape(self.batch_size, self.img_ch, np.shape(GT)[1], np.shape(GT)[2])

			#SR = F.sigmoid(self.unet(images))
			SR = torch.sigmoid(self.unet(images))
			acc += get_accuracy(SR,GT)
			SE += get_sensitivity(SR,GT)
			SP += get_specificity(SR,GT)
			PC += get_precision(SR,GT)
			F1 += get_F1(SR,GT)
			JS += get_JS(SR,GT)
			DC_RR += get_DC_RR(SR,GT, inverse_ratio = self.inverse_ratio)
			DC += get_DC(SR, GT)
			length += images.size(0)
			np_img = np.squeeze(SR.cpu().detach().numpy())
			np.save(self.result_img_path + str(i) + '.npy', np_img)
            
		acc = acc/length
		SE = SE/length
		SP = SP/length
		PC = PC/length
		F1 = F1/length
		JS = JS/length
		DC = DC/length
		DC_RR = DC_RR/length

		print('model type: ', self.model_type, 'accuracy: ', acc, 'sensitivity: ', SE, 'specificity: ', SP, 'precision: ', PC, 'F1 score: ', F1, 'Jaccard similarity: ', JS, 'Dice Coefficient: ', DC, 'DC_RR: ', DC_RR)
		result_csv_path = '/home/raphael/Projects/DL-Lung_Nodule_LUNA16/Solutions/RaphaelRosalie-solution/patch-based_U-net/results/'
		f = open(os.path.join(result_csv_path, 'result_compare.csv'), 'a', encoding = 'utf-8', newline= '')
		wr = csv.writer(f)
		wr.writerow([self.model_type, self.PPorLS, 'Accuracy: %.4f' % acc, 'Sensitivity: %.4f' % SE, 'Specificity: %.4f' % SP, 'Precision: %.4f'% PC, \
                            'F1 Score: %.4f' % F1, 'Jaccard Similarity: %.4f' % JS, 'Dice Coefficient: %.4f' % DC, 'RR_DC: %.4f' % DC_RR, 'inverse_ratio: %.3f' % self.inverse_ratio])
		f.close()
	        
        

	def whole_slice_prediction(self):
		"""Inference mode. Return whole slice prediction as a binary nodule mask."""
		unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%.4f-%s-%s.pkl' %(self.model_type,self.num_epochs,self.initial_lr,self.augmentation_prob, self.PPorLS, self.optimizer_choice))
		self.build_model()
		self.unet.load_state_dict(torch.load(unet_path))
		
		self.unet.train(False)
		self.unet.eval()
		acc = 0.	# Accuracy
		SE = 0.		# Sensitivity (Recall)
		SP = 0.		# Specificity
		PC = 0. 	# Precision
		F1 = 0.		# F1 Score
		JS = 0.		# Jaccard Similarity
		DC = 0.		# Dice Coefficient
		DC_RR = 0
		length=0
		for batch, (images, GT) in enumerate(self.whole_slice_prediction_loader):
			images = images.to(self.device)
			GT = GT.to(self.device)
			# Reshape the images and GT to 4-dimensional so that they can get fed to the conv2d layer.
			images = images.reshape(self.batch_size, self.img_ch, np.shape(images)[1], np.shape(images)[2])
			GT = GT.reshape(self.batch_size, self.img_ch, np.shape(GT)[1], np.shape(GT)[2])

			#SR = F.sigmoid(self.unet(images))
			SR = torch.sigmoid(self.unet(images))
			acc += get_accuracy(SR,GT)
			SE += get_sensitivity(SR,GT)
			SP += get_specificity(SR,GT)
			PC += get_precision(SR,GT)
			F1 += get_F1(SR,GT)
			JS += get_JS(SR,GT)
			DC_RR += get_DC_RR(SR,GT)
			DC += get_DC(SR, GT)
			length += images.size(0)
			np_img = np.squeeze(SR.cpu().detach().numpy())
			np.save(self.result_img_path + str(i) + '.npy', np_img)
            
            
		acc = acc/length
		SE = SE/length
		SP = SP/length
		PC = PC/length
		F1 = F1/length
		JS = JS/length
		DC = DC/length
		DC_RR = DC_RR/length
		unet_score = DC