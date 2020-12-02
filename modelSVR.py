import os
import time
import math
import random
import numpy as np
import h5py

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

import mcubes
from bspt import digest_bsp, get_mesh, get_mesh_watertight
#from bspt_slow import digest_bsp, get_mesh, get_mesh_watertight

from utils import *

#pytorch 1.2.0 implementation


class generator(nn.Module):
	def __init__(self, p_dim, c_dim):
		super(generator, self).__init__()
		self.p_dim = p_dim
		self.c_dim = c_dim
		convex_layer_weights = torch.zeros((self.p_dim, self.c_dim))
		self.convex_layer_weights = nn.Parameter(convex_layer_weights)
		nn.init.normal_(self.convex_layer_weights, mean=0.0, std=0.02)

	def forward(self, points, plane_m, convex_mask=None, is_training=False):
		#level 1
		h1 = torch.matmul(points, plane_m)
		h1 = torch.clamp(h1, min=0)

		#level 2
		h2 = torch.matmul(h1, (self.convex_layer_weights>0.01).float())

		#level 3
		if convex_mask is None:
			h3 = torch.min(h2, dim=2, keepdim=True)[0]
		else:
			h3 = torch.min(h2+convex_mask, dim=2, keepdim=True)[0]

		return h2,h3

class resnet_block(nn.Module):
	def __init__(self, dim_in, dim_out):
		super(resnet_block, self).__init__()
		self.dim_in = dim_in
		self.dim_out = dim_out
		if self.dim_in == self.dim_out:
			self.conv_1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=1, padding=1, bias=False)
			self.conv_2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=1, bias=False)
			nn.init.xavier_uniform_(self.conv_1.weight)
			nn.init.xavier_uniform_(self.conv_2.weight)
		else:
			self.conv_1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=2, padding=1, bias=False)
			self.conv_2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=1, bias=False)
			self.conv_s = nn.Conv2d(self.dim_in, self.dim_out, 1, stride=2, padding=0, bias=False)
			nn.init.xavier_uniform_(self.conv_1.weight)
			nn.init.xavier_uniform_(self.conv_2.weight)
			nn.init.xavier_uniform_(self.conv_s.weight)

	def forward(self, input, is_training=False):
		if self.dim_in == self.dim_out:
			output = self.conv_1(input)
			output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
			output = self.conv_2(output)
			output = output+input
			output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
		else:
			output = self.conv_1(input)
			output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
			output = self.conv_2(output)
			input_ = self.conv_s(input)
			output = output+input_
			output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
		return output

class img_encoder(nn.Module):
	def __init__(self, img_ef_dim, z_dim):
		super(img_encoder, self).__init__()
		self.img_ef_dim = img_ef_dim
		self.z_dim = z_dim
		self.conv_0 = nn.Conv2d(1, self.img_ef_dim, 7, stride=2, padding=3, bias=False)
		self.res_1 = resnet_block(self.img_ef_dim, self.img_ef_dim)
		self.res_2 = resnet_block(self.img_ef_dim, self.img_ef_dim)
		self.res_3 = resnet_block(self.img_ef_dim, self.img_ef_dim*2)
		self.res_4 = resnet_block(self.img_ef_dim*2, self.img_ef_dim*2)
		self.res_5 = resnet_block(self.img_ef_dim*2, self.img_ef_dim*4)
		self.res_6 = resnet_block(self.img_ef_dim*4, self.img_ef_dim*4)
		self.res_7 = resnet_block(self.img_ef_dim*4, self.img_ef_dim*8)
		self.res_8 = resnet_block(self.img_ef_dim*8, self.img_ef_dim*8)
		self.conv_9 = nn.Conv2d(self.img_ef_dim*8, self.img_ef_dim*16, 4, stride=2, padding=1, bias=True)
		self.conv_10 = nn.Conv2d(self.img_ef_dim*16, self.img_ef_dim*16, 4, stride=1, padding=0, bias=True)
		self.linear_1 = nn.Linear(self.img_ef_dim*16, self.img_ef_dim*16, bias=True)
		self.linear_2 = nn.Linear(self.img_ef_dim*16, self.img_ef_dim*16, bias=True)
		self.linear_3 = nn.Linear(self.img_ef_dim*16, self.img_ef_dim*16, bias=True)
		self.linear_4 = nn.Linear(self.img_ef_dim*16, self.z_dim, bias=True)
		nn.init.xavier_uniform_(self.conv_0.weight)
		nn.init.xavier_uniform_(self.conv_9.weight)
		nn.init.constant_(self.conv_9.bias,0)
		nn.init.xavier_uniform_(self.conv_10.weight)
		nn.init.constant_(self.conv_10.bias,0)
		nn.init.xavier_uniform_(self.linear_1.weight)
		nn.init.constant_(self.linear_1.bias,0)
		nn.init.xavier_uniform_(self.linear_2.weight)
		nn.init.constant_(self.linear_2.bias,0)
		nn.init.xavier_uniform_(self.linear_3.weight)
		nn.init.constant_(self.linear_3.bias,0)
		nn.init.xavier_uniform_(self.linear_4.weight)
		nn.init.constant_(self.linear_4.bias,0)

	def forward(self, view, is_training=False):
		layer_0 = self.conv_0(1-view)
		layer_0 = F.leaky_relu(layer_0, negative_slope=0.01, inplace=True)

		layer_1 = self.res_1(layer_0, is_training=is_training)
		layer_2 = self.res_2(layer_1, is_training=is_training)
		
		layer_3 = self.res_3(layer_2, is_training=is_training)
		layer_4 = self.res_4(layer_3, is_training=is_training)
		
		layer_5 = self.res_5(layer_4, is_training=is_training)
		layer_6 = self.res_6(layer_5, is_training=is_training)
		
		layer_7 = self.res_7(layer_6, is_training=is_training)
		layer_8 = self.res_8(layer_7, is_training=is_training)
		
		layer_9 = self.conv_9(layer_8)
		layer_9 = F.leaky_relu(layer_9, negative_slope=0.01, inplace=True)
		
		layer_10 = self.conv_10(layer_9)
		layer_10 = layer_10.view(-1,self.img_ef_dim*16)
		layer_10 = F.leaky_relu(layer_10, negative_slope=0.01, inplace=True)

		l1 = self.linear_1(layer_10)
		l1 = F.leaky_relu(l1, negative_slope=0.01, inplace=True)

		l2 = self.linear_2(l1)
		l2 = F.leaky_relu(l2, negative_slope=0.01, inplace=True)

		l3 = self.linear_3(l2)
		l3 = F.leaky_relu(l3, negative_slope=0.01, inplace=True)

		l4 = self.linear_4(l3)
		l4 = torch.sigmoid(l4)

		return l4

class decoder(nn.Module):
	def __init__(self, ef_dim, p_dim):
		super(decoder, self).__init__()
		self.ef_dim = ef_dim
		self.p_dim = p_dim
		self.linear_1 = nn.Linear(self.ef_dim*8, self.ef_dim*16, bias=True)
		self.linear_2 = nn.Linear(self.ef_dim*16, self.ef_dim*32, bias=True)
		self.linear_3 = nn.Linear(self.ef_dim*32, self.ef_dim*64, bias=True)
		self.linear_4 = nn.Linear(self.ef_dim*64, self.p_dim*4, bias=True)
		nn.init.xavier_uniform_(self.linear_1.weight)
		nn.init.constant_(self.linear_1.bias,0)
		nn.init.xavier_uniform_(self.linear_2.weight)
		nn.init.constant_(self.linear_2.bias,0)
		nn.init.xavier_uniform_(self.linear_3.weight)
		nn.init.constant_(self.linear_3.bias,0)
		nn.init.xavier_uniform_(self.linear_4.weight)
		nn.init.constant_(self.linear_4.bias,0)

	def forward(self, inputs, is_training=False):
		l1 = self.linear_1(inputs)
		l1 = F.leaky_relu(l1, negative_slope=0.01, inplace=True)

		l2 = self.linear_2(l1)
		l2 = F.leaky_relu(l2, negative_slope=0.01, inplace=True)

		l3 = self.linear_3(l2)
		l3 = F.leaky_relu(l3, negative_slope=0.01, inplace=True)

		l4 = self.linear_4(l3)
		l4 = l4.view(-1, 4, self.p_dim)

		return l4

class bsp_network(nn.Module):
	def __init__(self, ef_dim, p_dim, c_dim, img_ef_dim, z_dim):
		super(bsp_network, self).__init__()
		self.ef_dim = ef_dim
		self.p_dim = p_dim
		self.c_dim = c_dim
		self.img_ef_dim = img_ef_dim
		self.z_dim = z_dim
		self.img_encoder = img_encoder(self.img_ef_dim, self.z_dim)
		self.decoder = decoder(self.ef_dim, self.p_dim)
		self.generator = generator(self.p_dim, self.c_dim)

	def forward(self, inputs, z_vector, plane_m, point_coord, convex_mask=None, is_training=False):
		if is_training:
			z_vector = self.img_encoder(inputs, is_training=is_training)
			plane_m = None
			net_out_convexes = None
			net_out = None
		else:
			if inputs is not None:
				z_vector = self.img_encoder(inputs, is_training=is_training)
			if z_vector is not None:
				plane_m = self.decoder(z_vector, is_training=is_training)
			if point_coord is not None:
				net_out_convexes, net_out = self.generator(point_coord, plane_m, convex_mask=convex_mask, is_training=is_training)
			else:
				net_out_convexes = None
				net_out = None

		return z_vector, plane_m, net_out_convexes, net_out


class BSP_SVR(object):
	def __init__(self, config):
		"""
		Args:
			too lazy to explain
		"""
		self.input_size = 64 #input voxel grid size

		self.ef_dim = 32
		self.p_dim = 4096
		self.c_dim = 256
		self.img_ef_dim = 64
		self.z_dim = self.ef_dim*8

		#actual batch size
		self.shape_batch_size = 64

		self.view_size = 137
		self.crop_size = 128
		self.view_num = 24
		self.crop_edge = self.view_size-self.crop_size
		self.test_idx = 23

		self.dataset_name = config.dataset
		self.dataset_load = self.dataset_name + '_train'
		if not config.train:
			self.dataset_load = self.dataset_name + '_test'
		self.checkpoint_dir = config.checkpoint_dir
		self.data_dir = config.data_dir
		
		data_hdf5_name = self.data_dir+'/'+self.dataset_load+'.hdf5'
		if os.path.exists(data_hdf5_name):
			data_dict = h5py.File(data_hdf5_name, 'r')
			offset_x = int(self.crop_edge/2)
			offset_y = int(self.crop_edge/2)
			#reshape to NCHW
			self.data_pixels = np.reshape(data_dict['pixels'][:,:,offset_y:offset_y+self.crop_size, offset_x:offset_x+self.crop_size], [-1,self.view_num,1,self.crop_size,self.crop_size])
		else:
			print("error: cannot load "+data_hdf5_name)
			exit(0)
		if config.train:
			dataz_hdf5_name = self.checkpoint_dir+'/'+self.modelAE_dir+'/'+self.dataset_name+'_train_z.hdf5'
			if os.path.exists(dataz_hdf5_name):
				dataz_dict = h5py.File(dataz_hdf5_name, 'r')
				self.data_zs = dataz_dict['zs'][:]
			else:
				print("error: cannot load "+dataz_hdf5_name)
				exit(0)
			if len(self.data_zs) != len(self.data_pixels):
				print("error: len(self.data_zs) != len(self.data_pixels)")
				print(len(self.data_zs), len(self.data_pixels))
				exit(0)
		
		self.real_size = 64 #output point-value voxel grid size in testing
		self.test_size = 32 #related to testing batch_size, adjust according to gpu memory size
		test_point_batch_size = self.test_size*self.test_size*self.test_size #do not change
		
		#get coords
		dima = self.test_size
		dim = self.real_size
		self.aux_x = np.zeros([dima,dima,dima],np.uint8)
		self.aux_y = np.zeros([dima,dima,dima],np.uint8)
		self.aux_z = np.zeros([dima,dima,dima],np.uint8)
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier
		multiplier3 = multiplier*multiplier*multiplier
		for i in range(dima):
			for j in range(dima):
				for k in range(dima):
					self.aux_x[i,j,k] = i*multiplier
					self.aux_y[i,j,k] = j*multiplier
					self.aux_z[i,j,k] = k*multiplier
		self.coords = np.zeros([multiplier3,dima,dima,dima,3],np.float32)
		for i in range(multiplier):
			for j in range(multiplier):
				for k in range(multiplier):
					self.coords[i*multiplier2+j*multiplier+k,:,:,:,0] = self.aux_x+i
					self.coords[i*multiplier2+j*multiplier+k,:,:,:,1] = self.aux_y+j
					self.coords[i*multiplier2+j*multiplier+k,:,:,:,2] = self.aux_z+k
		self.coords = (self.coords+0.5)/dim-0.5
		self.coords = np.reshape(self.coords,[multiplier3,test_point_batch_size,3])
		self.coords = np.concatenate([self.coords, np.ones([multiplier3,test_point_batch_size,1],np.float32) ],axis=2)
		self.coords = torch.from_numpy(self.coords)

		if torch.cuda.is_available():
			self.device = torch.device('cuda')
			torch.backends.cudnn.benchmark = True
		else:
			self.device = torch.device('cpu')
		self.coords = self.coords.to(self.device)

		#build model
		self.bsp_network = bsp_network(self.ef_dim, self.p_dim, self.c_dim, self.img_ef_dim, self.z_dim)
		self.bsp_network.to(self.device)
		#print params
		#for param_tensor in self.bsp_network.state_dict():
		#	print(param_tensor, "\t", self.bsp_network.state_dict()[param_tensor].size())
		self.optimizer = torch.optim.Adam(self.bsp_network.img_encoder.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))
		#pytorch does not have a checkpoint manager
		#have to define it myself to manage max num of checkpoints to keep
		self.max_to_keep = 10
		self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir)
		self.checkpoint_name='BSP_SVR.model'
		self.checkpoint_manager_list = [None] * self.max_to_keep
		self.checkpoint_manager_pointer = 0
		self.checkpoint_AE_path = os.path.join(self.checkpoint_dir, self.modelAE_dir)
		self.checkpoint_AE_name='BSP_AE.model'
		#loss
		def network_loss(pred_z, gt_z):
			return torch.mean((pred_z - gt_z)**2)
		self.loss = network_loss

	@property
	def model_dir(self):
		return "{}_svr_{}".format(
				self.dataset_name, self.crop_size)
	@property
	def modelAE_dir(self):
		return "{}_ae_{}".format(
				self.dataset_name, self.input_size)

	def load(self):
		#load previous checkpoint
		checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
		if os.path.exists(checkpoint_txt):
			fin = open(checkpoint_txt)
			model_dir = fin.readline().strip()
			fin.close()
			self.bsp_network.load_state_dict(torch.load(model_dir))
			print(" [*] Load SUCCESS")
			return True
		else:
			print(" [!] Load failed...")
			return False

	def loadAE(self):
		#load AE weights
		checkpoint_txt = os.path.join(self.checkpoint_AE_path, "checkpoint")
		if os.path.exists(checkpoint_txt):
			fin = open(checkpoint_txt)
			model_dir = fin.readline().strip()
			fin.close()
			self.bsp_network.load_state_dict(torch.load(model_dir), strict=False)
			print(" [*] Load SUCCESS")
			return True
		else:
			print(" [!] Load failed...")
			return False

	def save(self,epoch):
		if not os.path.exists(self.checkpoint_path):
			os.makedirs(self.checkpoint_path)
		save_dir = os.path.join(self.checkpoint_path,self.checkpoint_name+"-"+str(epoch)+".pth")
		self.checkpoint_manager_pointer = (self.checkpoint_manager_pointer+1)%self.max_to_keep
		#delete checkpoint
		if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
			if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
				os.remove(self.checkpoint_manager_list[self.checkpoint_manager_pointer])
		#save checkpoint
		torch.save(self.bsp_network.state_dict(), save_dir)
		#update checkpoint manager
		self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir
		#write file
		checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
		fout = open(checkpoint_txt, 'w')
		for i in range(self.max_to_keep):
			pointer = (self.checkpoint_manager_pointer+self.max_to_keep-i)%self.max_to_keep
			if self.checkpoint_manager_list[pointer] is not None:
				fout.write(self.checkpoint_manager_list[pointer]+"\n")
		fout.close()


	def train(self, config):
		#load AE weights
		if not self.loadAE(): exit(-1)

		shape_num = len(self.data_pixels)
		batch_index_list = np.arange(shape_num)
		
		print("\n\n----------net summary----------")
		print("training samples   ", shape_num)
		print("-------------------------------\n\n")
		
		start_time = time.time()
		assert config.epoch==0 or config.iteration==0
		training_epoch = config.epoch + int(config.iteration/shape_num)
		batch_num = int(shape_num/self.shape_batch_size)
		#batch_view = np.zeros([self.shape_batch_size,self.crop_size,self.crop_size,1], np.float32)

		self.bsp_network.train()
		for epoch in range(0, training_epoch):
			np.random.shuffle(batch_index_list)
			avg_loss = 0
			avg_num = 0
			for idx in range(batch_num):
				dxb = batch_index_list[idx*self.shape_batch_size:(idx+1)*self.shape_batch_size]

				'''
				#random flip - not used
				for t in range(self.shape_batch_size):
					which_view = np.random.randint(self.view_num)
					batch_view_ = self.data_pixels[dxb[t],which_view].astype(np.float32)
					if np.random.randint(2)==0:
						batch_view_ = np.flip(batch_view_, 1)
					batch_view[t] = batch_view_/255.0
				'''
				
				which_view = np.random.randint(self.view_num)
				batch_view = self.data_pixels[dxb,which_view].astype(np.float32)/255.0
				batch_zs = self.data_zs[dxb]

				batch_view = torch.from_numpy(batch_view)
				batch_zs = torch.from_numpy(batch_zs)

				batch_view = batch_view.to(self.device)
				batch_zs = batch_zs.to(self.device)

				self.bsp_network.zero_grad()
				z_vector, _,_,_ = self.bsp_network(batch_view, None, None, None, is_training=True)
				err = self.loss(z_vector, batch_zs)

				err.backward()
				self.optimizer.step()

				avg_loss += err
				avg_num += 1
			print("Epoch: [%2d/%2d] time: %4.4f, loss: %.8f" % (epoch, training_epoch, time.time() - start_time, avg_loss/avg_num))
			if epoch%10==9:
				self.test_1(config,"train_"+str(epoch))
			if epoch%100==99:
				self.save(epoch)

		self.save(training_epoch)


	def test_1(self, config, name):
		multiplier = int(self.real_size/self.test_size)
		multiplier2 = multiplier*multiplier

		thres = 0.99
	
		t = np.random.randint(len(self.data_pixels))
		model_float = np.zeros([self.real_size+2,self.real_size+2,self.real_size+2],np.float32)
		batch_view = self.data_pixels[t:t+1,self.test_idx].astype(np.float32)/255.0
		batch_view = torch.from_numpy(batch_view)
		batch_view = batch_view.to(self.device)
		_, out_m, _,_ = self.bsp_network(batch_view, None, None, None, is_training=False)
		for i in range(multiplier):
			for j in range(multiplier):
				for k in range(multiplier):
					minib = i*multiplier2+j*multiplier+k
					point_coord = self.coords[minib:minib+1]
					_,_,_, net_out = self.bsp_network(None, None, out_m, point_coord, is_training=False)
					net_out = torch.clamp(1-net_out, min=0, max=1)
					model_float[self.aux_x+i+1,self.aux_y+j+1,self.aux_z+k+1] = np.reshape(net_out.detach().cpu().numpy(), [self.test_size,self.test_size,self.test_size])
		
		vertices, triangles = mcubes.marching_cubes(model_float, thres)
		vertices = (vertices-0.5)/self.real_size-0.5
		#output ply sum
		write_ply_triangle(config.sample_dir+"/"+name+".ply", vertices, triangles)
		print("[sample]")


	#output bsp shape as ply
	def test_bsp(self, config):
		#load previous checkpoint
		if not self.load(): exit(-1)
		
		w2 = self.bsp_network.generator.convex_layer_weights.detach().cpu().numpy()

		dima = self.test_size
		dim = self.real_size
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier

		self.bsp_network.eval()
		for t in range(config.start, min(len(self.data_pixels),config.end)):
			model_float = np.ones([self.real_size,self.real_size,self.real_size,self.c_dim],np.float32)
			batch_view = self.data_pixels[t:t+1,self.test_idx].astype(np.float32)/255.0
			batch_view = torch.from_numpy(batch_view)
			batch_view = batch_view.to(self.device)
			_, out_m, _,_ = self.bsp_network(batch_view, None, None, None, is_training=False)
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						minib = i*multiplier2+j*multiplier+k
						point_coord = self.coords[minib:minib+1]
						_,_, model_out, _ = self.bsp_network(None, None, out_m, point_coord, is_training=False)
						model_float[self.aux_x+i,self.aux_y+j,self.aux_z+k,:] = np.reshape(model_out.detach().cpu().numpy(), [self.test_size,self.test_size,self.test_size,self.c_dim])
			
			out_m = out_m.detach().cpu().numpy()
			
			bsp_convex_list = []
			model_float = model_float<0.01
			model_float_sum = np.sum(model_float,axis=3)
			for i in range(self.c_dim):
				slice_i = model_float[:,:,:,i]
				if np.max(slice_i)>0: #if one voxel is inside a convex
					if np.min(model_float_sum-slice_i*2)>=0: #if this convex is redundant, i.e. the convex is inside the shape
						model_float_sum = model_float_sum-slice_i
					else:
						box = []
						for j in range(self.p_dim):
							if w2[j,i]>0.01:
								a = -out_m[0,0,j]
								b = -out_m[0,1,j]
								c = -out_m[0,2,j]
								d = -out_m[0,3,j]
								box.append([a,b,c,d])
						if len(box)>0:
							bsp_convex_list.append(np.array(box,np.float32))

			#print(bsp_convex_list)
			print(len(bsp_convex_list))
			
			#convert bspt to mesh
			#vertices, polygons = get_mesh(bsp_convex_list)
			#use the following alternative to merge nearby vertices to get watertight meshes
			vertices, polygons = get_mesh_watertight(bsp_convex_list)

			#output ply
			write_ply_polygon(config.sample_dir+"/"+str(t)+"_bsp.ply", vertices, polygons)
	
	#output bsp shape as ply and point cloud as ply
	def test_mesh_point(self, config):
		#load previous checkpoint
		if not self.load(): exit(-1)

		w2 = self.bsp_network.generator.convex_layer_weights.detach().cpu().numpy()
		dima = self.test_size
		dim = self.real_size
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier

		self.bsp_network.eval()
		for t in range(config.start, min(len(self.data_pixels),config.end)):
			print(t)
			model_float = np.ones([self.real_size,self.real_size,self.real_size,self.c_dim],np.float32)
			model_float_combined = np.ones([self.real_size,self.real_size,self.real_size],np.float32)
			batch_view = self.data_pixels[t:t+1,self.test_idx].astype(np.float32)/255.0
			batch_view = torch.from_numpy(batch_view)
			batch_view = batch_view.to(self.device)
			_, out_m, _,_ = self.bsp_network(batch_view, None, None, None, is_training=False)
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						minib = i*multiplier2+j*multiplier+k
						point_coord = self.coords[minib:minib+1]
						_,_, model_out, model_out_combined = self.bsp_network(None, None, out_m, point_coord, is_training=False)
						model_float[self.aux_x+i,self.aux_y+j,self.aux_z+k,:] = np.reshape(model_out.detach().cpu().numpy(), [self.test_size,self.test_size,self.test_size,self.c_dim])
						model_float_combined[self.aux_x+i,self.aux_y+j,self.aux_z+k] = np.reshape(model_out_combined.detach().cpu().numpy(), [self.test_size,self.test_size,self.test_size])
			
			out_m_ = out_m.detach().cpu().numpy()

			# whether to use post processing to remove convexes that are inside the shape
			post_processing_flag = False
			
			if post_processing_flag:
				bsp_convex_list = []
				model_float = model_float<0.01
				model_float_sum = np.sum(model_float,axis=3)
				unused_convex = np.ones([self.c_dim], np.float32)
				for i in range(self.c_dim):
					slice_i = model_float[:,:,:,i]
					if np.max(slice_i)>0: #if one voxel is inside a convex
						if np.min(model_float_sum-slice_i*2)>=0: #if this convex is redundant, i.e. the convex is inside the shape
							model_float_sum = model_float_sum-slice_i
						else:
							box = []
							for j in range(self.p_dim):
								if w2[j,i]>0.01:
									a = -out_m_[0,0,j]
									b = -out_m_[0,1,j]
									c = -out_m_[0,2,j]
									d = -out_m_[0,3,j]
									box.append([a,b,c,d])
							if len(box)>0:
								bsp_convex_list.append(np.array(box,np.float32))
								unused_convex[i] = 0
								
				#convert bspt to mesh
				#vertices, polygons = get_mesh(bsp_convex_list)
				#use the following alternative to merge nearby vertices to get watertight meshes
				vertices, polygons = get_mesh_watertight(bsp_convex_list)

				#output ply
				write_ply_polygon(config.sample_dir+"/"+str(t)+"_bsp.ply", vertices, polygons)
				#output obj
				#write_obj_polygon(config.sample_dir+"/"+str(t)+"_bsp.obj", vertices, polygons)
				
				#sample surface points
				sampled_points_normals = sample_points_polygon(vertices, polygons, 16384)
				#check point inside shape or not
				point_coord = np.reshape(sampled_points_normals[:,:3]+sampled_points_normals[:,3:]*1e-4, [1,-1,3])
				point_coord = np.concatenate([point_coord, np.ones([1,point_coord.shape[1],1],np.float32) ],axis=2)
				_,_,_, sample_points_value = self.bsp_network(None, None, out_m, torch.from_numpy(point_coord).to(self.device), convex_mask=torch.from_numpy(np.reshape(unused_convex, [1,1,-1])).to(self.device), is_training=False)
				sample_points_value = sample_points_value.detach().cpu().numpy()
				sampled_points_normals = sampled_points_normals[sample_points_value[0,:,0]>1e-4]
				print(len(bsp_convex_list), len(sampled_points_normals))
				np.random.shuffle(sampled_points_normals)
				write_ply_point_normal(config.sample_dir+"/"+str(t)+"_pc.ply", sampled_points_normals[:4096])
			else:
				bsp_convex_list = []
				model_float = model_float<0.01
				model_float_sum = np.sum(model_float,axis=3)
				for i in range(self.c_dim):
					slice_i = model_float[:,:,:,i]
					if np.max(slice_i)>0: #if one voxel is inside a convex
						#if np.min(model_float_sum-slice_i*2)>=0: #if this convex is redundant, i.e. the convex is inside the shape
						#	model_float_sum = model_float_sum-slice_i
						#else:
							box = []
							for j in range(self.p_dim):
								if w2[j,i]>0.01:
									a = -out_m_[0,0,j]
									b = -out_m_[0,1,j]
									c = -out_m_[0,2,j]
									d = -out_m_[0,3,j]
									box.append([a,b,c,d])
							if len(box)>0:
								bsp_convex_list.append(np.array(box,np.float32))
								
				#convert bspt to mesh
				#vertices, polygons = get_mesh(bsp_convex_list)
				#use the following alternative to merge nearby vertices to get watertight meshes
				vertices, polygons = get_mesh_watertight(bsp_convex_list)

				#output ply
				write_ply_polygon(config.sample_dir+"/"+str(t)+"_bsp.ply", vertices, polygons)
				#output obj
				#write_obj_polygon(config.sample_dir+"/"+str(t)+"_bsp.obj", vertices, polygons)
				
				#sample surface points
				sampled_points_normals = sample_points_polygon_vox64(vertices, polygons, model_float_combined, 16384)
				#check point inside shape or not
				point_coord = np.reshape(sampled_points_normals[:,:3]+sampled_points_normals[:,3:]*1e-4, [1,-1,3])
				point_coord = np.concatenate([point_coord, np.ones([1,point_coord.shape[1],1],np.float32) ],axis=2)
				_,_,_, sample_points_value = self.bsp_network(None, None, out_m, torch.from_numpy(point_coord).to(self.device), is_training=False)
				sample_points_value = sample_points_value.detach().cpu().numpy()
				sampled_points_normals = sampled_points_normals[sample_points_value[0,:,0]>1e-4]
				print(len(bsp_convex_list), len(sampled_points_normals))
				np.random.shuffle(sampled_points_normals)
				write_ply_point_normal(config.sample_dir+"/"+str(t)+"_pc.ply", sampled_points_normals[:4096])


	#output bsp shape as obj with color
	def test_mesh_obj_material(self, config):
		#load previous checkpoint
		if not self.load(): exit(-1)
		
		w2 = self.bsp_network.generator.convex_layer_weights.detach().cpu().numpy()

		dima = self.test_size
		dim = self.real_size
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier

		#write material
		#all output shapes share the same material
		#which means the same convex always has the same color for different shapes
		#change the colors in default.mtl to visualize correspondences between shapes
		fout2 = open(config.sample_dir+"/default.mtl", 'w')
		for i in range(self.c_dim):
			fout2.write("newmtl m"+str(i+1)+"\n") #material id
			fout2.write("Kd 0.80 0.80 0.80\n") #color (diffuse) RGB 0.00-1.00
			fout2.write("Ka 0 0 0\n") #color (ambient) leave 0s
		fout2.close()

		self.bsp_network.eval()
		for t in range(config.start, min(len(self.data_pixels),config.end)):
			model_float = np.ones([self.real_size,self.real_size,self.real_size,self.c_dim],np.float32)
			batch_view = self.data_pixels[t:t+1,self.test_idx].astype(np.float32)/255.0
			batch_view = torch.from_numpy(batch_view)
			batch_view = batch_view.to(self.device)
			_, out_m, _,_ = self.bsp_network(batch_view, None, None, None, is_training=False)
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						minib = i*multiplier2+j*multiplier+k
						point_coord = self.coords[minib:minib+1]
						_,_, model_out, _ = self.bsp_network(None, None, out_m, point_coord, is_training=False)
						model_float[self.aux_x+i,self.aux_y+j,self.aux_z+k,:] = np.reshape(model_out.detach().cpu().numpy(), [self.test_size,self.test_size,self.test_size,self.c_dim])
			
			out_m = out_m.detach().cpu().numpy()
			
			bsp_convex_list = []
			color_idx_list = []
			model_float = model_float<0.01
			model_float_sum = np.sum(model_float,axis=3)
			for i in range(self.c_dim):
				slice_i = model_float[:,:,:,i]
				if np.max(slice_i)>0: #if one voxel is inside a convex
					if np.min(model_float_sum-slice_i*2)>=0: #if this convex is redundant, i.e. the convex is inside the shape
						model_float_sum = model_float_sum-slice_i
					else:
						box = []
						for j in range(self.p_dim):
							if w2[j,i]>0.01:
								a = -out_m[0,0,j]
								b = -out_m[0,1,j]
								c = -out_m[0,2,j]
								d = -out_m[0,3,j]
								box.append([a,b,c,d])
						if len(box)>0:
							bsp_convex_list.append(np.array(box,np.float32))
							color_idx_list.append(i)

			#print(bsp_convex_list)
			print(len(bsp_convex_list))
			
			#convert bspt to mesh
			vertices = []

			#write obj
			fout2 = open(config.sample_dir+"/"+str(t)+"_bsp.obj", 'w')
			fout2.write("mtllib default.mtl\n")

			for i in range(len(bsp_convex_list)):
				vg, tg = get_mesh([bsp_convex_list[i]])
				vbias=len(vertices)+1
				vertices = vertices+vg

				fout2.write("usemtl m"+str(color_idx_list[i]+1)+"\n")
				for ii in range(len(vg)):
					fout2.write("v "+str(vg[ii][0])+" "+str(vg[ii][1])+" "+str(vg[ii][2])+"\n")
				for ii in range(len(tg)):
					fout2.write("f")
					for jj in range(len(tg[ii])):
						fout2.write(" "+str(tg[ii][jj]+vbias))
					fout2.write("\n")

			fout2.close()
