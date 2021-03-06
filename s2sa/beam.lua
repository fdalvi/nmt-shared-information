require 'nn'
require 'string'
require 'nngraph'

--require 'cutorch'
--require 'cunn'

require 's2sa.models'
require 's2sa.data'
require 's2sa.scorer'

path = require 'pl.path'
stringx = require 'pl.stringx'

local sent_id = 0
local opt = {}
local cmd = torch.CmdLine()

-- file location
cmd:option('-model', 'seq2seq_lstm_attn.t7.', [[Path to model .t7 file]])
cmd:option('-extract', 'enc', [[For encoding generation, the nmt module to extract (enc/dec)]])
cmd:option('-extract_layer', -1, [[For encoding generation, the encoding layer to extract]])
cmd:option('-src_file', '', [[Source sequence to decode (one line per sequence)]])
cmd:option('-targ_file', '', [[True target sequence (optional)]])
cmd:option('-output_file', 'pred.txt', [[Path to output the predictions (each line will be the decoded sequence]])
cmd:option('-src_dict', 'data/demo.src.dict', [[Path to source vocabulary (*.src.dict file)]])
cmd:option('-targ_dict', 'data/demo.targ.dict', [[Path to target vocabulary (*.targ.dict file)]])
cmd:option('-feature_dict_prefix', 'data/demo', [[Prefix of the path to features vocabularies (*.feature_N.dict files)]])
cmd:option('-char_dict', 'data/demo.char.dict', [[If using chars, path to character vocabulary (*.char.dict file)]])

-- beam search options
cmd:option('-beam', 5, [[Beam size]])
cmd:option('-max_sent_l', 250, [[Maximum sentence length. If any sequences in srcfile are longer than this then it will error out]])
cmd:option('-simple', 0, [[If = 1, output prediction is simply the first time the top of the beam
                         ends with an end-of-sentence token. If = 0, the model considers all
                         hypotheses that have been generated so far that ends with end-of-sentence
                         token and takes the highest scoring of all of them.]])
cmd:option('-replace_unk', 0, [[Replace the generated UNK tokens with the source token that
                              had the highest attention weight. If srctarg_dict is provided,
                              it will lookup the identified source token and give the corresponding
                              target token. If it is not provided (or the identified source token
                              does not exist in the table) then it will copy the source token]])
cmd:option('-srctarg_dict', 'data/en-de.dict', [[Path to source-target dictionary to replace UNK
                                               tokens. See README.md for the format this file should be in]])
cmd:option('-score_gold', 0, [[If = 1, score the log likelihood of the gold as well]])
cmd:option('-n_best', 1, [[If > 1, it will also output an n_best list of decoded sentences]])
cmd:option('-gpuid', -1, [[ID of the GPU to use (-1 = use CPU)]])
cmd:option('-gpuid2', -1, [[Second GPU ID]])
cmd:option('-cudnn', 0, [[If using character model, this should be = 1 if the character model was trained using cudnn]])

cmd:option('-rescore', '', [[use specified metric to select best translation in the beam, available: bleu, gleu]])
cmd:option('-rescore_param', 4, [[parameter for the scoring metric, for BLEU is corresponding to n_gram ]])

cmd:option('-mask', '', [[specify mask in the format #neuron,#neuron,#neuron....]])

function copy(orig)
  local orig_type = type(orig)
  local copy
  if orig_type == 'table' then
    copy = {}
    for orig_key, orig_value in pairs(orig) do
      copy[orig_key] = orig_value
    end
  else
    copy = orig
  end
  return copy
end

local function append_table(dst, src)
  for i = 1, #src do
    table.insert(dst, src[i])
  end
end

local StateAll = torch.class("StateAll")

function StateAll.initial(start)
  return {start}
end

function StateAll.advance(state, token)
  local new_state = copy(state)
  table.insert(new_state, token)
  return new_state
end

function StateAll.disallow(out)
  local bad = {1, 3} -- 1 is PAD, 3 is BOS
  for j = 1, #bad do
    out[bad[j]] = -1e9
  end
end

function StateAll.same(state1, state2)
  for i = 2, #state1 do
    if state1[i] ~= state2[i] then
      return false
    end
  end
  return true
end

function StateAll.next(state)
  return state[#state]
end

function StateAll.heuristic(state)
  return 0
end

function StateAll.print(state)
  for i = 1, #state do
    io.write(state[i] .. " ")
  end
  print()
end

-- Convert a flat index to a row-column tuple.
function flat_to_rc(v, flat_index)
  local row = math.floor((flat_index - 1) / v:size(2)) + 1
  return row, (flat_index - 1) % v:size(2) + 1
end

function generate_beam(model, initial, K, max_sent_l, source, source_features, gold)
  --reset decoder initial states
  if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
    cutorch.setDevice(opt.gpuid)
  end
  local n = max_sent_l
  -- Backpointer table.
  local prev_ks = torch.LongTensor(n, K):fill(1)
  -- Current States.
  local next_ys = torch.LongTensor(n, K):fill(1)
  -- Current Scores.
  local scores = torch.FloatTensor(n, K)
  scores:zero()
  local source_l = math.min(source:size(1), opt.max_sent_l)
  local attn_argmax = {} -- store attn weights
  attn_argmax[1] = {}

  local states = {} -- store predicted word idx
  states[1] = {}
  for k = 1, 1 do
    table.insert(states[1], initial)
    table.insert(attn_argmax[1], initial)
    next_ys[1][k] = State.next(initial)
  end

  local source_input
  if model_opt.use_chars_enc == 1 then
    source_input = source:view(source_l, 1, source:size(2)):contiguous()
  else
    source_input = source:view(source_l, 1)
  end

  local rnn_state_enc = {}
  for i = 1, #init_fwd_enc do
    table.insert(rnn_state_enc, init_fwd_enc[i]:zero())
  end
  local context = context_proto[{{}, {1,source_l}}]:clone() -- 1 x source_l x rnn_size

  for t = 1, source_l do
    local encoder_input = {source_input[t]}
    if model_opt.num_source_features > 0 then
      append_table(encoder_input, source_features[t])
    end
    append_table(encoder_input, rnn_state_enc)
    local out = model[1]:forward(encoder_input)
    rnn_state_enc = out
    context[{{},t}]:copy(out[#out])
  end
  rnn_state_dec = {}
  for i = 1, #init_fwd_dec do
    table.insert(rnn_state_dec, init_fwd_dec[i]:zero())
  end

  if model_opt.init_dec == 1 then
    for L = 1, model_opt.num_layers do
      rnn_state_dec[L*2-1+model_opt.input_feed]:copy(
        rnn_state_enc[L*2-1]:expand(K, model_opt.rnn_size))
      rnn_state_dec[L*2+model_opt.input_feed]:copy(
        rnn_state_enc[L*2]:expand(K, model_opt.rnn_size))
    end
  end

  if model_opt.brnn == 1 then
    for i = 1, #rnn_state_enc do
      rnn_state_enc[i]:zero()
    end
    for t = source_l, 1, -1 do
      local encoder_input = {source_input[t]}
      if model_opt.num_source_features > 0 then
        append_table(encoder_input, source_features[t])
      end
      append_table(encoder_input, rnn_state_enc)
      local out = model[4]:forward(encoder_input)
      rnn_state_enc = out
      context[{{},t}]:add(out[#out])
    end
    if model_opt.init_dec == 1 then
      for L = 1, model_opt.num_layers do
        rnn_state_dec[L*2-1+model_opt.input_feed]:add(
          rnn_state_enc[L*2-1]:expand(K, model_opt.rnn_size))
        rnn_state_dec[L*2+model_opt.input_feed]:add(
          rnn_state_enc[L*2]:expand(K, model_opt.rnn_size))
      end
    end
  end
  if opt.score_gold == 1 then
    rnn_state_dec_gold = {}
    for i = 1, #rnn_state_dec do
      table.insert(rnn_state_dec_gold, rnn_state_dec[i][{{1}}]:clone())
    end
  end

  context = context:expand(K, source_l, model_opt.rnn_size)

  if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
    cutorch.setDevice(opt.gpuid2)
    local context2 = context_proto2[{{1, K}, {1, source_l}}]
    context2:copy(context)
    context = context2
  end

  out_float = torch.FloatTensor()

  local i = 1
  local done = false
  local max_score = -1e9
  local found_eos = false

  while (not done) and (i < n) do
    i = i+1
    states[i] = {}
    attn_argmax[i] = {}
    local decoder_input1
    if model_opt.use_chars_dec == 1 then
      decoder_input1 = word2charidx_targ:index(1, next_ys:narrow(1,i-1,1):squeeze())
    else
      decoder_input1 = next_ys:narrow(1,i-1,1):squeeze()
      if opt.beam == 1 then
        decoder_input1 = torch.LongTensor({decoder_input1})
      end
    end
    local decoder_input
    if model_opt.attn == 1 then
      decoder_input = {decoder_input1, context, table.unpack(rnn_state_dec)}
    else
      decoder_input = {decoder_input1, context[{{}, source_l}], table.unpack(rnn_state_dec)}
    end
    local out_decoder = model[2]:forward(decoder_input)
    local out = model[3]:forward(out_decoder[#out_decoder]) -- K x vocab_size

    rnn_state_dec = {} -- to be modified later
    if model_opt.input_feed == 1 then
      table.insert(rnn_state_dec, out_decoder[#out_decoder])
    end
    for j = 1, #out_decoder - 1 do
      table.insert(rnn_state_dec, out_decoder[j])
    end
    out_float:resize(out:size()):copy(out)
    for k = 1, K do
      State.disallow(out_float:select(1, k))
      out_float[k]:add(scores[i-1][k])
    end
    -- All the scores available.

    local flat_out = out_float:view(-1)
    if i == 2 then
      flat_out = out_float[1] -- all outputs same for first batch
    end

    if model_opt.start_symbol == 1 then
      decoder_softmax.output[{{},1}]:zero()
      decoder_softmax.output[{{},source_l}]:zero()
    end

    for k = 1, K do
      while true do
        local score, index = flat_out:max(1)
        local score = score[1]
        local prev_k, y_i = flat_to_rc(out_float, index[1])
        states[i][k] = State.advance(states[i-1][prev_k], y_i)
        local diff = true
        for k2 = 1, k-1 do
          if State.same(states[i][k2], states[i][k]) then
            diff = false
          end
        end

        if i < 2 or diff then
          if model_opt.attn == 1 then
            max_attn, max_index = decoder_softmax.output[prev_k]:max(1)
            attn_argmax[i][k] = State.advance(attn_argmax[i-1][prev_k],max_index[1])
          end
          prev_ks[i][k] = prev_k
          next_ys[i][k] = y_i
          scores[i][k] = score
          flat_out[index[1]] = -1e9
          break -- move on to next k
        end
        flat_out[index[1]] = -1e9
      end
    end
    for j = 1, #rnn_state_dec do
      rnn_state_dec[j]:copy(rnn_state_dec[j]:index(1, prev_ks[i]))
    end
    end_hyp = states[i][1]
    end_score = scores[i][1]
    if model_opt.attn == 1 then
      end_attn_argmax = attn_argmax[i][1]
    end
    if end_hyp[#end_hyp] == END then
      done = true
      found_eos = true
    else
      for k = 1, K do
        local possible_hyp = states[i][k]
        if possible_hyp[#possible_hyp] == END then
          found_eos = true
          if scores[i][k] > max_score then
            max_hyp = possible_hyp
            max_score = scores[i][k]
            if model_opt.attn == 1 then
              max_attn_argmax = attn_argmax[i][k]
            end
          end
        end
      end
    end
  end

  local best_mscore = -1e9
  local mscore_hyp
  local mscore_scores
  local mscore_attn_argmax
  local gold_table
  if opt.rescore ~= '' then
    gold_table = gold[{{2, gold:size(1)-1}}]:totable()
    for k = 1, K do
        local hyp={}
        for _,v in pairs(states[i][k]) do
          if v == END then break; end
          table.insert(hyp,v)
        end
        table.insert(hyp, END)
        local score_k = scorers[opt.rescore](hyp, gold_table, opt.rescore_param)
        if score_k > best_mscore then
          mscore_hyp = hyp
          mscore_scores = scores[i][k]
          best_mscore = score_k
          mscore_attn_argmax = attn_argmax[i][k]
        end
    end
  end

  local gold_score = 0
  if opt.score_gold == 1 then
    rnn_state_dec = {}
    for i = 1, #init_fwd_dec do
      table.insert(rnn_state_dec, init_fwd_dec[i][{{1}}]:zero())
    end
    if model_opt.init_dec == 1 then
      rnn_state_dec = rnn_state_dec_gold
    end
    local target_l = gold:size(1)
    for t = 2, target_l do
      local decoder_input1
      if model_opt.use_chars_dec == 1 then
        decoder_input1 = word2charidx_targ:index(1, gold[{{t-1}}])
      else
        decoder_input1 = gold[{{t-1}}]
      end
      local decoder_input
      if model_opt.attn == 1 then
        decoder_input = {decoder_input1, context[{{1}}], table.unpack(rnn_state_dec)}
      else
        decoder_input = {decoder_input1, context[{{1}, source_l}], table.unpack(rnn_state_dec)}
      end
      local out_decoder = model[2]:forward(decoder_input)
      local out = model[3]:forward(out_decoder[#out_decoder]) -- K x vocab_size
      rnn_state_dec = {} -- to be modified later
      if model_opt.input_feed == 1 then
        table.insert(rnn_state_dec, out_decoder[#out_decoder])
      end
      for j = 1, #out_decoder - 1 do
        table.insert(rnn_state_dec, out_decoder[j])
      end
      gold_score = gold_score + out[1][gold[t]]
    end
  end
  if opt.simple == 1 or end_score > max_score or not found_eos then
    max_hyp = end_hyp
    max_score = end_score
    max_attn_argmax = end_attn_argmax
  end

  local best_hyp=states[i]
  local best_scores=scores[i]
  local best_attn_argmax=attn_argmax[i]
  if opt.rescore ~= '' then
    local max_mscore = scorers[opt.rescore](max_hyp, gold_table, opt.rescore_param)
    print('RESCORE MAX '..opt.rescore..': '..max_mscore, 'BEST '..opt.rescore..': '..best_mscore)
    max_hyp=mscore_hyp
    max_score=best_mscore
    max_attn_argmax=mscore_attn_argmax
    best_hyp=mscore_hyp
    best_scores=mscore_scores
    best_attn_argmax=mscore_attn_argmax
  end

  return max_hyp, max_score, max_attn_argmax, gold_score, best_hyp, best_scores, best_attn_argmax
end


function generate_beam_with_mask(model, initial, K, max_sent_l, source, source_features, gold, mask)
  --reset decoder initial states
  if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
    cutorch.setDevice(opt.gpuid)
  end
  local n = max_sent_l
  -- Backpointer table.
  local prev_ks = torch.LongTensor(n, K):fill(1)
  -- Current States.
  local next_ys = torch.LongTensor(n, K):fill(1)
  -- Current Scores.
  local scores = torch.FloatTensor(n, K)
  scores:zero()
  local source_l = math.min(source:size(1), opt.max_sent_l)
  local attn_argmax = {} -- store attn weights
  attn_argmax[1] = {}

  local states = {} -- store predicted word idx
  states[1] = {}
  for k = 1, 1 do
    table.insert(states[1], initial)
    table.insert(attn_argmax[1], initial)
    next_ys[1][k] = State.next(initial)
  end

  local source_input
  if model_opt.use_chars_enc == 1 then
    source_input = source:view(source_l, 1, source:size(2)):contiguous()
  else
    source_input = source:view(source_l, 1)
  end
  -- print(source_input)
  local rnn_state_enc = {}
  for i = 1, #init_fwd_enc do
    table.insert(rnn_state_enc, init_fwd_enc[i]:zero())
  end
  local context = context_proto[{{}, {1,source_l}}]:clone() -- 1 x source_l x rnn_size

  for t = 1, source_l do
    local encoder_input = {source_input[t]}
    if model_opt.num_source_features > 0 then
      append_table(encoder_input, source_features[t])
    end
    append_table(encoder_input, rnn_state_enc)
    
    --print('=========================')
    --for i = 1, #encoder_input do
    --  print(encoder_input[i]:sum())
    --end
    --print('--------')
    --local out = model[1]:forward(encoder_input)
    -- print('---------------------')
    
    local temp_input = encoder_input
    
    -- 1: identity layer
    local moduleNum = 1
    -- print(string.format("%d: %s", moduleNum, model[1].modules[moduleNum]))
    temp_input = model[1].modules[moduleNum]:forward(temp_input)
    moduleNum = moduleNum + 1
    
    -- Now layerwise decoding
    local offset = 0
    local outputs = {}
    for L = 1, model_opt.num_layers do
        local prev_c = temp_input[L*2+offset]
        local prev_h = temp_input[L*2+1+offset]
        
        local x
        if L == 1 then
          -- 2: Word Embedding layer
          -- print(string.format("%d: %s", moduleNum, model[1].modules[moduleNum]))
          x = model[1].modules[moduleNum]:forward(temp_input[1])
          moduleNum = moduleNum + 1
        else
          x = outputs[(L-1)*2]
          if model_opt.dropout > 0 then
            --print(string.format("%d: %s", moduleNum, model[1].modules[moduleNum]))
            --print("dropout")
            x = model[1].modules[moduleNum]:forward(x)
            moduleNum = moduleNum + 1
          end
        end
        
        -- print(string.format("%d: %s", moduleNum, model[1].modules[moduleNum]))
        local i2h = model[1].modules[moduleNum]:forward(x)
        moduleNum = moduleNum + 1
        
        --ignore identity
        moduleNum = moduleNum + 1
        
        -- print(string.format("%d: %s", moduleNum, model[1].modules[moduleNum]))
        local h2h = model[1].modules[moduleNum]:forward(prev_h)
        moduleNum = moduleNum + 1
        
        -- print(string.format("%d: %s", moduleNum, model[1].modules[moduleNum]))
        local all_input_sums = model[1].modules[moduleNum]:forward({i2h, h2h})
        moduleNum = moduleNum + 1
        
        -- print(string.format("%d: %s", moduleNum, model[1].modules[moduleNum]))
        local reshaped = model[1].modules[moduleNum]:forward(all_input_sums)
        moduleNum = moduleNum + 1
        
        -- print(string.format("%d: %s", moduleNum, model[1].modules[moduleNum]))
        local n1, n2, n3, n4 = unpack(model[1].modules[moduleNum]:forward(reshaped))
        moduleNum = moduleNum + 1
        
        -- print(string.format("%d: %s", moduleNum, model[1].modules[moduleNum]))
        local forget_gate = model[1].modules[moduleNum]:forward(n2)
        moduleNum = moduleNum + 1
        
        --ignore identity
        moduleNum = moduleNum + 1
        
        -- print(string.format("%d: %s", moduleNum, model[1].modules[moduleNum]))
        -- print("next_c_1")
        local next_c_1 = model[1].modules[moduleNum]:forward({forget_gate, prev_c})
        moduleNum = moduleNum + 1
        
        -- print(string.format("%d: %s", moduleNum, model[1].modules[moduleNum]))
        local in_gate = model[1].modules[moduleNum]:forward(n1)
        moduleNum = moduleNum + 1
        
        -- print(string.format("%d: %s", moduleNum, model[1].modules[moduleNum]))
        local in_transform = model[1].modules[moduleNum]:forward(n4)
        moduleNum = moduleNum + 1
        
        -- print(string.format("%d: %s", moduleNum, model[1].modules[moduleNum]))
        -- print("next_c_2")
        local next_c_2 = model[1].modules[moduleNum]:forward({in_gate, in_transform})
        moduleNum = moduleNum + 1
        
        -- print(string.format("%d: %s", moduleNum, model[1].modules[moduleNum]))
        -- print("next_c")
        local next_c = model[1].modules[moduleNum]:forward({next_c_1, next_c_2})
        moduleNum = moduleNum + 1
        
        -- print(string.format("%d: %s", moduleNum, model[1].modules[moduleNum]))
        local out_gate = model[1].modules[moduleNum]:forward(n3)
        moduleNum = moduleNum + 1
        
        -- print(string.format("%d: %s", moduleNum, model[1].modules[moduleNum]))
        local tanh_temp = model[1].modules[moduleNum]:forward(next_c)
        moduleNum = moduleNum + 1
        
        -- print(string.format("%d: %s", moduleNum, model[1].modules[moduleNum]))
        local next_h = model[1].modules[moduleNum]:forward({out_gate, tanh_temp})
        moduleNum = moduleNum + 1
        
        -- mask outputs
        -- print('F', L, next_h:sum())
        next_h[mask[L]] = 0
        -- print('F', L, next_h:sum())
        
        table.insert(outputs, next_c)
        table.insert(outputs, next_h)
    end
    -- for i,module in ipairs(model[1]:listModules()) do
    --   -- print(module)
    -- end
    
    --assert(#out == #outputs)
    --for i = 1, #outputs do
    --  assert(torch.all(torch.eq(out[i], outputs[i])))
    --  print(torch.sum(outputs[i]))
    --end
    --print('---------------------')
    rnn_state_enc = outputs
    context[{{},t}]:copy(outputs[#outputs])
  end
  rnn_state_dec = {}
  for i = 1, #init_fwd_dec do
    table.insert(rnn_state_dec, init_fwd_dec[i]:zero())
  end

  if model_opt.init_dec == 1 then
    for L = 1, model_opt.num_layers do
      rnn_state_dec[L*2-1+model_opt.input_feed]:copy(
        rnn_state_enc[L*2-1]:expand(K, model_opt.rnn_size))
      rnn_state_dec[L*2+model_opt.input_feed]:copy(
        rnn_state_enc[L*2]:expand(K, model_opt.rnn_size))
    end
  end
  if model_opt.brnn == 1 then
    for i = 1, #rnn_state_enc do
      rnn_state_enc[i]:zero()
    end
    for t = source_l, 1, -1 do
      local encoder_input = {source_input[t]}
      if model_opt.num_source_features > 0 then
        append_table(encoder_input, source_features[t])
      end
      append_table(encoder_input, rnn_state_enc)
      -- local out = model[4]:forward(encoder_input)
      
      ------------------------------------------------------------------------------
      local temp_input = encoder_input
      --print('===brnn======================')
      --for i = 1, #encoder_input do
      --  print(encoder_input[i]:sum())
      --end
      --print('---brnn-----')
    
      -- 1: identity layer
      local moduleNum = 1
      -- print(string.format("%d: %s", moduleNum, model[4].modules[moduleNum]))
      temp_input = model[4].modules[moduleNum]:forward(temp_input)
      moduleNum = moduleNum + 1
      
      -- Now layerwise decoding
      local offset = 0
      local outputs = {}
      for L = 1, model_opt.num_layers do
          local prev_c = temp_input[L*2+offset]
          local prev_h = temp_input[L*2+1+offset]
          
          local x
          if L == 1 then
            -- 2: Word Embedding layer
            -- print(string.format("%d: %s", moduleNum, model[4].modules[moduleNum]))
            x = model[4].modules[moduleNum]:forward(temp_input[1])
            moduleNum = moduleNum + 1
          else
            x = outputs[(L-1)*2]
            if model_opt.dropout > 0 then
              -- print(string.format("%d: %s", moduleNum, model[4].modules[moduleNum]))
              x = model[4].modules[moduleNum]:forward(x)
              moduleNum = moduleNum + 1
            end
          end
          
          -- print(string.format("%d: %s", moduleNum, model[4].modules[moduleNum]))
          local i2h = model[4].modules[moduleNum]:forward(x)
          moduleNum = moduleNum + 1
          
          --ignore identity
          moduleNum = moduleNum + 1
          
          -- print(string.format("%d: %s", moduleNum, model[4].modules[moduleNum]))
          local h2h = model[4].modules[moduleNum]:forward(prev_h)
          moduleNum = moduleNum + 1
          
          -- print(string.format("%d: %s", moduleNum, model[4].modules[moduleNum]))
          local all_input_sums = model[4].modules[moduleNum]:forward({i2h, h2h})
          moduleNum = moduleNum + 1
          
          -- print(string.format("%d: %s", moduleNum, model[4].modules[moduleNum]))
          local reshaped = model[4].modules[moduleNum]:forward(all_input_sums)
          moduleNum = moduleNum + 1
          
          -- print(string.format("%d: %s", moduleNum, model[4].modules[moduleNum]))
          local n1, n2, n3, n4 = unpack(model[4].modules[moduleNum]:forward(reshaped))
          moduleNum = moduleNum + 1
          
          -- print(string.format("%d: %s", moduleNum, model[4].modules[moduleNum]))
          local forget_gate = model[4].modules[moduleNum]:forward(n2)
          moduleNum = moduleNum + 1
          
          --ignore identity
          moduleNum = moduleNum + 1
          
          -- print(string.format("%d: %s", moduleNum, model[4].modules[moduleNum]))
          -- print("next_c_1")
          local next_c_1 = model[4].modules[moduleNum]:forward({forget_gate, prev_c})
          moduleNum = moduleNum + 1
          
          -- print(string.format("%d: %s", moduleNum, model[4].modules[moduleNum]))
          local in_gate = model[4].modules[moduleNum]:forward(n1)
          moduleNum = moduleNum + 1
          
          -- print(string.format("%d: %s", moduleNum, model[4].modules[moduleNum]))
          local in_transform = model[4].modules[moduleNum]:forward(n4)
          moduleNum = moduleNum + 1
          
          -- print(string.format("%d: %s", moduleNum, model[4].modules[moduleNum]))
          -- print("next_c_2")
          local next_c_2 = model[4].modules[moduleNum]:forward({in_gate, in_transform})
          moduleNum = moduleNum + 1
          
          -- print(string.format("%d: %s", moduleNum, model[4].modules[moduleNum]))
          -- print("next_c")
          local next_c = model[4].modules[moduleNum]:forward({next_c_1, next_c_2})
          moduleNum = moduleNum + 1
          
          -- print(string.format("%d: %s", moduleNum, model[4].modules[moduleNum]))
          local out_gate = model[4].modules[moduleNum]:forward(n3)
          moduleNum = moduleNum + 1
          
          -- print(string.format("%d: %s", moduleNum, model[4].modules[moduleNum]))
          local tanh_temp = model[4].modules[moduleNum]:forward(next_c)
          moduleNum = moduleNum + 1
          
          -- print(string.format("%d: %s", moduleNum, model[4].modules[moduleNum]))
          local next_h = model[4].modules[moduleNum]:forward({out_gate, tanh_temp})
          moduleNum = moduleNum + 1
          
          -- mask outputs
          -- print('B', L, next_h:sum())
          next_h[mask[model_opt.num_layers + L]] = 0
          -- print('B', L, next_h:sum())
          
          table.insert(outputs, next_c)
          table.insert(outputs, next_h)
      end
      ------------------------------------------------------------------------------

      -- assert(#out == #outputs)
      --for i = 1, #outputs do
        -- assert(torch.all(torch.eq(out[i], outputs[i])))
      --  print(outputs[i]:sum())
      --end
      --print('---------------------')
      
      rnn_state_enc = outputs
      context[{{},t}]:add(outputs[#outputs])
    end
    if model_opt.init_dec == 1 then
      for L = 1, model_opt.num_layers do
        rnn_state_dec[L*2-1+model_opt.input_feed]:add(
          rnn_state_enc[L*2-1]:expand(K, model_opt.rnn_size))
        rnn_state_dec[L*2+model_opt.input_feed]:add(
          rnn_state_enc[L*2]:expand(K, model_opt.rnn_size))
      end
    end
  end
  if opt.score_gold == 1 then
    rnn_state_dec_gold = {}
    for i = 1, #rnn_state_dec do
      table.insert(rnn_state_dec_gold, rnn_state_dec[i][{{1}}]:clone())
    end
  end

  context = context:expand(K, source_l, model_opt.rnn_size)

  if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
    cutorch.setDevice(opt.gpuid2)
    local context2 = context_proto2[{{1, K}, {1, source_l}}]
    context2:copy(context)
    context = context2
  end

  out_float = torch.FloatTensor()

  local i = 1
  local done = false
  local max_score = -1e9
  local found_eos = false

  while (not done) and (i < n) do
    i = i+1
    states[i] = {}
    attn_argmax[i] = {}
    local decoder_input1
    if model_opt.use_chars_dec == 1 then
      decoder_input1 = word2charidx_targ:index(1, next_ys:narrow(1,i-1,1):squeeze())
    else
      decoder_input1 = next_ys:narrow(1,i-1,1):squeeze()
      if opt.beam == 1 then
        decoder_input1 = torch.LongTensor({decoder_input1})
      end
    end
    local decoder_input
    if model_opt.attn == 1 then
      decoder_input = {decoder_input1, context, table.unpack(rnn_state_dec)}
    else
      decoder_input = {decoder_input1, context[{{}, source_l}], table.unpack(rnn_state_dec)}
    end
    local out_decoder = model[2]:forward(decoder_input)
    local out = model[3]:forward(out_decoder[#out_decoder]) -- K x vocab_size

    rnn_state_dec = {} -- to be modified later
    if model_opt.input_feed == 1 then
      table.insert(rnn_state_dec, out_decoder[#out_decoder])
    end
    for j = 1, #out_decoder - 1 do
      table.insert(rnn_state_dec, out_decoder[j])
    end
    out_float:resize(out:size()):copy(out)
    for k = 1, K do
      State.disallow(out_float:select(1, k))
      out_float[k]:add(scores[i-1][k])
    end
    -- All the scores available.

    local flat_out = out_float:view(-1)
    if i == 2 then
      flat_out = out_float[1] -- all outputs same for first batch
    end

    if model_opt.start_symbol == 1 then
      decoder_softmax.output[{{},1}]:zero()
      decoder_softmax.output[{{},source_l}]:zero()
    end

    for k = 1, K do
      while true do
        local score, index = flat_out:max(1)
        local score = score[1]
        local prev_k, y_i = flat_to_rc(out_float, index[1])
        states[i][k] = State.advance(states[i-1][prev_k], y_i)
        local diff = true
        for k2 = 1, k-1 do
          if State.same(states[i][k2], states[i][k]) then
            diff = false
          end
        end

        if i < 2 or diff then
          if model_opt.attn == 1 then
            max_attn, max_index = decoder_softmax.output[prev_k]:max(1)
            attn_argmax[i][k] = State.advance(attn_argmax[i-1][prev_k],max_index[1])
          end
          prev_ks[i][k] = prev_k
          next_ys[i][k] = y_i
          scores[i][k] = score
          flat_out[index[1]] = -1e9
          break -- move on to next k
        end
        flat_out[index[1]] = -1e9
      end
    end
    for j = 1, #rnn_state_dec do
      rnn_state_dec[j]:copy(rnn_state_dec[j]:index(1, prev_ks[i]))
    end
    end_hyp = states[i][1]
    end_score = scores[i][1]
    if model_opt.attn == 1 then
      end_attn_argmax = attn_argmax[i][1]
    end
    if end_hyp[#end_hyp] == END then
      done = true
      found_eos = true
    else
      for k = 1, K do
        local possible_hyp = states[i][k]
        if possible_hyp[#possible_hyp] == END then
          found_eos = true
          if scores[i][k] > max_score then
            max_hyp = possible_hyp
            max_score = scores[i][k]
            if model_opt.attn == 1 then
              max_attn_argmax = attn_argmax[i][k]
            end
          end
        end
      end
    end
  end

  local best_mscore = -1e9
  local mscore_hyp
  local mscore_scores
  local mscore_attn_argmax
  local gold_table
  if opt.rescore ~= '' then
    gold_table = gold[{{2, gold:size(1)-1}}]:totable()
    for k = 1, K do
        local hyp={}
        for _,v in pairs(states[i][k]) do
          if v == END then break; end
          table.insert(hyp,v)
        end
        table.insert(hyp, END)
        local score_k = scorers[opt.rescore](hyp, gold_table, opt.rescore_param)
        if score_k > best_mscore then
          mscore_hyp = hyp
          mscore_scores = scores[i][k]
          best_mscore = score_k
          mscore_attn_argmax = attn_argmax[i][k]
        end
    end
  end

  local gold_score = 0
  if opt.score_gold == 1 then
    rnn_state_dec = {}
    for i = 1, #init_fwd_dec do
      table.insert(rnn_state_dec, init_fwd_dec[i][{{1}}]:zero())
    end
    if model_opt.init_dec == 1 then
      rnn_state_dec = rnn_state_dec_gold
    end
    local target_l = gold:size(1)
    for t = 2, target_l do
      local decoder_input1
      if model_opt.use_chars_dec == 1 then
        decoder_input1 = word2charidx_targ:index(1, gold[{{t-1}}])
      else
        decoder_input1 = gold[{{t-1}}]
      end
      local decoder_input
      if model_opt.attn == 1 then
        decoder_input = {decoder_input1, context[{{1}}], table.unpack(rnn_state_dec)}
      else
        decoder_input = {decoder_input1, context[{{1}, source_l}], table.unpack(rnn_state_dec)}
      end
      local out_decoder = model[2]:forward(decoder_input)
      local out = model[3]:forward(out_decoder[#out_decoder]) -- K x vocab_size
      rnn_state_dec = {} -- to be modified later
      if model_opt.input_feed == 1 then
        table.insert(rnn_state_dec, out_decoder[#out_decoder])
      end
      for j = 1, #out_decoder - 1 do
        table.insert(rnn_state_dec, out_decoder[j])
      end
      gold_score = gold_score + out[1][gold[t]]
    end
  end
  if opt.simple == 1 or end_score > max_score or not found_eos then
    max_hyp = end_hyp
    max_score = end_score
    max_attn_argmax = end_attn_argmax
  end

  local best_hyp=states[i]
  local best_scores=scores[i]
  local best_attn_argmax=attn_argmax[i]
  if opt.rescore ~= '' then
    local max_mscore = scorers[opt.rescore](max_hyp, gold_table, opt.rescore_param)
    print('RESCORE MAX '..opt.rescore..': '..max_mscore, 'BEST '..opt.rescore..': '..best_mscore)
    max_hyp=mscore_hyp
    max_score=best_mscore
    max_attn_argmax=mscore_attn_argmax
    best_hyp=mscore_hyp
    best_scores=mscore_scores
    best_attn_argmax=mscore_attn_argmax
  end

  return max_hyp, max_score, max_attn_argmax, gold_score, best_hyp, best_scores, best_attn_argmax
end


function idx2key(file)
  local f = io.open(file,'r')
  local t = {}
  for line in f:lines() do
    local c = {}
    for w in line:gmatch'([^%s]+)' do
      table.insert(c, w)
    end
    t[tonumber(c[2])] = c[1]
  end
  return t
end

function flip_table(u)
  local t = {}
  for key, value in pairs(u) do
    t[value] = key
  end
  return t
end

function get_layer(layer)
  if layer.name ~= nil then
    if layer.name == 'decoder_attn' then
      decoder_attn = layer
    elseif layer.name:sub(1,3) == 'hop' then
      hop_attn = layer
    elseif layer.name:sub(1,7) == 'softmax' then
      table.insert(softmax_layers, layer)
    elseif layer.name == 'word_vecs_enc' then
      word_vecs_enc = layer
    elseif layer.name == 'word_vecs_dec' then
      word_vecs_dec = layer
    end
  end
end

local function features2featureidx(features, feature2idx, start_symbol)
  local out = {}

  if start_symbol == 1 then
    table.insert(out, {})
    for j = 1, #feature2idx do
      table.insert(out[#out], torch.Tensor(1):fill(START))
    end
  end

  for i = 1, #features do
    table.insert(out, {})
    for j = 1, #feature2idx do
      local value = feature2idx[j][features[i][j]]
      if value == nil then
        value = UNK
      end
      table.insert(out[#out], torch.Tensor(1):fill(value))
    end
  end

  if start_symbol == 1 then
    table.insert(out, {})
    for j = 1, #feature2idx do
      table.insert(out[#out], torch.Tensor(1):fill(END))
    end
  end

  return out
end

function sent2wordidx(sent, word2idx, start_symbol)
  local t = {}
  local u = {}
  if start_symbol == 1 then
    table.insert(t, START)
    table.insert(u, START_WORD)
  end

  for word in sent:gmatch'([^%s]+)' do
    local idx = word2idx[word] or UNK
    table.insert(t, idx)
    table.insert(u, word)
  end
  if start_symbol == 1 then
    table.insert(t, END)
    table.insert(u, END_WORD)
  end
  return torch.LongTensor(t), u
end

function sent2charidx(sent, char2idx, max_word_l, start_symbol)
  local words = {}
  if start_symbol == 1 then
    table.insert(words, START_WORD)
  end
  for word in sent:gmatch'([^%s]+)' do
    table.insert(words, word)
  end
  if start_symbol == 1 then
    table.insert(words, END_WORD)
  end
  local chars = torch.ones(#words, max_word_l)
  for i = 1, #words do
    chars[i] = word2charidx(words[i], char2idx, max_word_l, chars[i])
  end
  return chars, words
end

function word2charidx(word, char2idx, max_word_l, t)
  t[1] = START
  local i = 2
  for _, char in utf8.next, word do
    char = utf8.char(char)
    local char_idx = char2idx[char] or UNK
    t[i] = char_idx
    i = i+1
    if i >= max_word_l then
      t[i] = END
      break
    end
  end
  if i < max_word_l then
    t[i] = END
  end
  return t
end

function wordidx2sent(sent, idx2word, source_str, attn, skip_end)
  local t = {}
  local start_i, end_i
  skip_end = skip_start_end or true
  if skip_end then
    end_i = #sent-1
  else
    end_i = #sent
  end
  for i = 2, end_i do -- skip START and END
    if sent[i] == UNK then
      if opt.replace_unk == 1 then
        local s = source_str[attn[i]]
        if phrase_table[s] ~= nil then
          print(s .. ':' ..phrase_table[s])
        end
        local r = phrase_table[s] or s
        table.insert(t, r)
      else
        table.insert(t, idx2word[sent[i]])
      end
    else
      table.insert(t, idx2word[sent[i]])
    end
  end
  return table.concat(t, ' ')
end

function clean_sent(sent)
  local s = stringx.replace(sent, UNK_WORD, '')
  s = stringx.replace(s, START_WORD, '')
  s = stringx.replace(s, END_WORD, '')
  s = stringx.replace(s, START_CHAR, '')
  s = stringx.replace(s, END_CHAR, '')
  return s
end

function strip(s)
  return s:gsub("^%s+",""):gsub("%s+$","")
end

local function extract_features(line)
  local cleaned_tokens = {}
  local features = {}

  for entry in line:gmatch'([^%s]+)' do
    local field = entry:split('%-|%-')
    local word = clean_sent(field[1])
    if string.len(word) > 0 then
      table.insert(cleaned_tokens, word)

      if #field > 1 then
        table.insert(features, {})
      end

      for i= 2, #field do
        table.insert(features[#features], field[i])
      end
    end
  end

  return cleaned_tokens, features
end

function init(arg)
  -- parse input params
  opt = cmd:parse(arg)

  -- some globals
  PAD = 1; UNK = 2; START = 3; END = 4
  PAD_WORD = '<blank>'; UNK_WORD = '<unk>'; START_WORD = '<s>'; END_WORD = '</s>'
  START_CHAR = '{'; END_CHAR = '}'
  MAX_SENT_L = opt.max_sent_l

  assert(path.exists(opt.model), 'model does not exist')

  if opt.gpuid >= 0 then
    require 'cutorch'
    require 'cunn'
    if opt.cudnn == 1 then
      require 'cudnn'
    end
  end
  print('loading ' .. opt.model .. '...')
  checkpoint = torch.load(opt.model)
  print('done!')

  if opt.replace_unk == 1 then
    phrase_table = {}
    if path.exists(opt.srctarg_dict) then
      local f = io.open(opt.srctarg_dict,'r')
      for line in f:lines() do
        local c = line:split("|||")
        phrase_table[strip(c[1])] = c[2]
      end
    end
  end

  if opt.rescore ~= '' then
    require 's2sa.scorer'
    if not scorers[opt.rescore] then
      error("metric "..opt.rescore.." not defined")
    end
  end

  -- load model and word2idx/idx2word dictionaries
  model, model_opt = checkpoint[1], checkpoint[2]
  
  opt.model_opt = model_opt
  for i = 1, #model do
    model[i]:cuda() -- remove for compatability if needed
    model[i]:evaluate()
  end
  -- for backward compatibility
  model_opt.brnn = model_opt.brnn or 0
  model_opt.input_feed = model_opt.input_feed or 1
  model_opt.attn = model_opt.attn or 1
  model_opt.num_source_features = model_opt.num_source_features or 0

  idx2word_src = idx2key(opt.src_dict)
  word2idx_src = flip_table(idx2word_src)
  idx2word_targ = idx2key(opt.targ_dict)
  word2idx_targ = flip_table(idx2word_targ)

  idx2feature_src = {}
  feature2idx_src = {}

  for i = 1, model_opt.num_source_features do
    table.insert(idx2feature_src, idx2key(opt.feature_dict_prefix .. '.source_feature_' .. i .. '.dict'))
    table.insert(feature2idx_src, flip_table(idx2feature_src[i]))
  end

  -- load character dictionaries if needed
  if model_opt.use_chars_enc == 1 or model_opt.use_chars_dec == 1 then
    utf8 = require 'lua-utf8'
    char2idx = flip_table(idx2key(opt.char_dict))
    model[1]:apply(get_layer)
  end
  if model_opt.use_chars_dec == 1 then
    word2charidx_targ = torch.LongTensor(#idx2word_targ, model_opt.max_word_l):fill(PAD)
    for i = 1, #idx2word_targ do
      word2charidx_targ[i] = word2charidx(idx2word_targ[i], char2idx,
        model_opt.max_word_l, word2charidx_targ[i])
    end
  end
  -- load gold labels if it exists
  if path.exists(opt.targ_file) then
    print('loading GOLD labels at ' .. opt.targ_file)
    gold = {}
    local file = io.open(opt.targ_file, 'r')
    for line in file:lines() do
      table.insert(gold, line)
    end
  else
    if opt.score_gold == 1 then
      error('missing targ_file option to calculate gold scores')
    end
  end

  if opt.gpuid >= 0 then
    cutorch.setDevice(opt.gpuid)
    for i = 1, #model do
      if opt.gpuid2 >= 0 then
        if i == 1 or i == 4 then
          cutorch.setDevice(opt.gpuid)
        else
          cutorch.setDevice(opt.gpuid2)
        end
      end
      model[i]:double():cuda()
      model[i]:evaluate()
    end
  end

  softmax_layers = {}
  model[2]:apply(get_layer)
  if model_opt.attn == 1 then
    decoder_attn:apply(get_layer)
    decoder_softmax = softmax_layers[1]
    attn_layer = torch.zeros(opt.beam, MAX_SENT_L)
  end

  context_proto = torch.zeros(1, MAX_SENT_L, model_opt.rnn_size)
  local h_init_dec = torch.zeros(opt.beam, model_opt.rnn_size)
  local h_init_enc = torch.zeros(1, model_opt.rnn_size)
  if opt.gpuid >= 0 then
    h_init_enc = h_init_enc:cuda()
    h_init_dec = h_init_dec:cuda()
    cutorch.setDevice(opt.gpuid)
    if opt.gpuid2 >= 0 then
      cutorch.setDevice(opt.gpuid)
      context_proto = context_proto:cuda()
      cutorch.setDevice(opt.gpuid2)
      context_proto2 = torch.zeros(opt.beam, MAX_SENT_L, model_opt.rnn_size):cuda()
    else
      context_proto = context_proto:cuda()
    end
    if model_opt.attn == 1 then
      attn_layer = attn_layer:cuda()
    end
  end
  init_fwd_enc = {}
  init_fwd_dec = {} -- initial context
  if model_opt.input_feed == 1 then
    table.insert(init_fwd_dec, h_init_dec:clone())
  end

  for L = 1, model_opt.num_layers do
    table.insert(init_fwd_enc, h_init_enc:clone())
    table.insert(init_fwd_enc, h_init_enc:clone())
    table.insert(init_fwd_dec, h_init_dec:clone()) -- memory cell
    table.insert(init_fwd_dec, h_init_dec:clone()) -- hidden state
  end

  pred_score_total = 0
  gold_score_total = 0
  pred_words_total = 0
  gold_words_total = 0

  State = StateAll
  sent_id = 0
end

function search(line)
  sent_id = sent_id + 1
  local cleaned_tokens, source_features_str = extract_features(line)
  local cleaned_line = table.concat(cleaned_tokens, ' ')
  print('SENT ' .. sent_id .. ': ' ..line)
  local source, source_str
  local source_features = features2featureidx(source_features_str, feature2idx_src, model_opt.start_symbol)
  if model_opt.use_chars_enc == 0 then
    source, source_str = sent2wordidx(cleaned_line, word2idx_src, model_opt.start_symbol)
  else
    source, source_str = sent2charidx(cleaned_line, char2idx, model_opt.max_word_l, model_opt.start_symbol)
  end
  if gold then
    target, target_str = sent2wordidx(gold[sent_id], word2idx_targ, 1)
  end
  state = State.initial(START)
  pred, pred_score, attn, gold_score, all_sents, all_scores, all_attn = generate_beam(model,
    state, opt.beam, MAX_SENT_L, source, source_features, target)
  pred_score_total = pred_score_total + pred_score
  pred_words_total = pred_words_total + #pred - 1
  pred_sent = wordidx2sent(pred, idx2word_targ, source_str, attn, true)

  print('PRED ' .. sent_id .. ': ' .. pred_sent)
  
  if gold ~= nil then
    print('GOLD ' .. sent_id .. ': ' .. gold[sent_id])
    if opt.score_gold == 1 then
      print(string.format("PRED SCORE: %.4f, GOLD SCORE: %.4f", pred_score, gold_score))
      gold_score_total = gold_score_total + gold_score
      gold_words_total = gold_words_total + target:size(1) - 1
    end
  end

  nbests = {}

  if opt.n_best > 1 then
    for n = 1, opt.n_best do
      pred_sent_n = wordidx2sent(all_sents[n], idx2word_targ, source_str, all_attn[n], false)
      local out_n = string.format("%d ||| %s ||| %.4f", n, pred_sent_n, all_scores[n])
      print(out_n)
      nbests[n] = out_n
    end
  end

  print('')

  return pred_sent, nbests
end

function search_with_mask(line, mask)
  sent_id = sent_id + 1
  local cleaned_tokens, source_features_str = extract_features(line)
  local cleaned_line = table.concat(cleaned_tokens, ' ')
  print('SENT ' .. sent_id .. ': ' ..line)
  local source, source_str
  local source_features = features2featureidx(source_features_str, feature2idx_src, model_opt.start_symbol)
  if model_opt.use_chars_enc == 0 then
    source, source_str = sent2wordidx(cleaned_line, word2idx_src, model_opt.start_symbol)
  else
    source, source_str = sent2charidx(cleaned_line, char2idx, model_opt.max_word_l, model_opt.start_symbol)
  end
  if gold then
    target, target_str = sent2wordidx(gold[sent_id], word2idx_targ, 1)
  end
  
  state = State.initial(START)
  pred, pred_score, attn, gold_score, all_sents, all_scores, all_attn = generate_beam_with_mask(model,
    state, opt.beam, MAX_SENT_L, source, source_features, target, mask)
  pred_score_total = pred_score_total + pred_score
  pred_words_total = pred_words_total + #pred - 1
  pred_sent = wordidx2sent(pred, idx2word_targ, source_str, attn, true)

  print('PRED ' .. sent_id .. ': ' .. pred_sent)
  
  if gold ~= nil then
    print('GOLD ' .. sent_id .. ': ' .. gold[sent_id])
    if opt.score_gold == 1 then
      print(string.format("PRED SCORE: %.4f, GOLD SCORE: %.4f", pred_score, gold_score))
      gold_score_total = gold_score_total + gold_score
      gold_words_total = gold_words_total + target:size(1) - 1
    end
  end

  nbests = {}

  if opt.n_best > 1 then
    for n = 1, opt.n_best do
      pred_sent_n = wordidx2sent(all_sents[n], idx2word_targ, source_str, all_attn[n], false)
      local out_n = string.format("%d ||| %s ||| %.4f", n, pred_sent_n, all_scores[n])
      print(out_n)
      nbests[n] = out_n
    end
  end

  print('')

  return pred_sent, nbests
end

-- Encode: take a seq2seq-attn model and a piece of text (line) and generate
-- a (1 x length in tokens x encoding size) matrix of the encodings for each token in this
-- piece of text.
--
-- This will be used for computing statistics like correlation between components of the encoding
-- over the space of inputs.
function encode(line, layer)
  -- Get tokens from string
  local cleaned_tokens, source_features_str = extract_features(line)
  local cleaned_line = table.concat(cleaned_tokens, ' ')

  -- print('SENT: ' ..line)

  -- Create the source vectors
  local source, source_str
  local source_features = features2featureidx(source_features_str, feature2idx_src, model_opt.start_symbol)
  if model_opt.use_chars_enc == 0 then
    source, source_str = sent2wordidx(cleaned_line, word2idx_src, model_opt.start_symbol)
  else
    source, source_str = sent2charidx(cleaned_line, char2idx, model_opt.max_word_l, model_opt.start_symbol)
  end

  -- Use the correct gpu if specified
  if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
    cutorch.setDevice(opt.gpuid)
  end

  -- Cap source length
  local source_l
  if source:size(1) > opt.max_sent_l then
    return nil, nil
  else
    source_l = source:size(1)
  end

  -- Format source input for encoder
  local source_input
  if model_opt.use_chars_enc == 1 then
    source_input = source:view(source_l, 1, source:size(2)):contiguous():cuda()
  else
    source_input = source:view(source_l, 1):cuda()
  end

  -- Initialize the encoder RNN state
  local rnn_state_enc = {}
  for i = 1, #init_fwd_enc do
    table.insert(rnn_state_enc, init_fwd_enc[i]:zero():cuda())
  end

  local context_size = model_opt.rnn_size

  -- Increase context_size if we care about all layers
  if layer == nil or layer < 0 then 
    context_size = context_size * model_opt.num_layers
  end

  local context = torch.zeros(1, source_l, context_size)

  -- We will also keep track of bwd_context in case of brnn model
  local backward_context
  if model_opt.brnn == 1 then
    backward_context = torch.zeros(1, source_l, context_size)
  end

  -- Iterate through the source
  for t = 1, source_l do
    -- Input this token
    local encoder_input = {source_input[t]}
    -- Add any auxiliary features provided
    if model_opt.num_source_features > 0 then
      append_table(encoder_input, source_features[t])
    end
    -- Add all of the pieces of RNN state
    append_table(encoder_input, rnn_state_enc) 

    -- Run one layer forward
    local out = model[1]:forward(encoder_input)
    rnn_state_enc = out

    local final_res
    if layer == nil or layer < 0 then
      -- By default combine all layers
      hidden_states = {}
      for h = 1, #out/2 do
        table.insert(hidden_states, out[h*2])
      end
      final_res = torch.cat(hidden_states)
    else
      final_res = out[layer]
    end 
    context[{{},t}]:copy(final_res)
  end

  -- Handle bidirectional rnn's
  if model_opt.brnn == 1 then
    for i = 1, #rnn_state_enc do
      rnn_state_enc[i]:zero()
    end
    for t = source_l, 1, -1 do
      local encoder_input = {source_input[t]}
      if model_opt.num_source_features > 0 then
        append_table(encoder_input, source_features[t])
      end
      append_table(encoder_input, rnn_state_enc)
      local out = model[4]:forward(encoder_input)
      rnn_state_enc = out

      local final_res
      if layer == nil or layer < 0 then
        -- By default combine all layers
        hidden_states = {}
        for h = 1, #out/2 do
          table.insert(hidden_states, out[h*2])
        end
        final_res = torch.cat(hidden_states)
      else
        final_res = out[layer*2]
      end 
      backward_context[{{},t}]:copy(final_res)
    end
  end

  -- combined fwd and bwd contexts
  if model_opt.brnn == 1 then
    context = torch.cat({context, backward_context}, 3)
  end

  return context
end

function decode(source_line, target_line, layer)
  -- Not using `encode` here for greater efficiency
  if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
    cutorch.setDevice(opt.gpuid)
  end

  -- source
  local cleaned_tokens, source_features_str = extract_features(source_line)
  local cleaned_line = table.concat(cleaned_tokens, ' ')

  -- Create the source vectors
  local source, source_str
  local source_features = features2featureidx(source_features_str, feature2idx_src, model_opt.start_symbol)
  if model_opt.use_chars_enc == 0 then
    source, source_str = sent2wordidx(cleaned_line, word2idx_src, model_opt.start_symbol)
  else
    source, source_str = sent2charidx(cleaned_line, char2idx, model_opt.max_word_l, model_opt.start_symbol)
  end

  -- Create target vector
  target, target_str = sent2wordidx(target_line, word2idx_targ, 1)

  -- Cap source length
  -- local source_l
  -- if source:size(1) > opt.max_sent_l then
  --   return nil
  -- else
  --   source_l = source:size(1)
  -- end
  local source_l = math.min(source:size(1), opt.max_sent_l)
  source = source[{{1,source_l}}]

  -- Cap target length
  local target_l
  if target:size(1) > opt.max_sent_l then
    return nil
  else
    target_l = target:size(1)
  end

  local source_input
  if model_opt.use_chars_enc == 1 then
    source_input = source:view(source_l, 1, source:size(2)):contiguous()
  else
    source_input = source:view(source_l, 1)
  end

  local rnn_state_enc = {}
  for i = 1, #init_fwd_enc do
    table.insert(rnn_state_enc, init_fwd_enc[i]:zero())
  end
  local context = context_proto[{{}, {1,source_l}}]:clone() -- 1 x source_l x rnn_size

  for t = 1, source_l do
    local encoder_input = {source_input[t]}
    append_table(encoder_input, rnn_state_enc)
    local out = model[1]:forward(encoder_input)
    rnn_state_enc = out
    context[{{},t}]:copy(out[#out])
  end
  rnn_state_dec = {}

  for i = 1, #init_fwd_dec do
    table.insert(rnn_state_dec, init_fwd_dec[i][{{1}}]:zero())
  end

  if model_opt.init_dec == 1 then
    for L = 1, model_opt.num_layers do
      rnn_state_dec[L*2-1+model_opt.input_feed]:copy(rnn_state_enc[L*2-1])
      rnn_state_dec[L*2+model_opt.input_feed]:copy(rnn_state_enc[L*2])
    end
  end

  if model_opt.brnn == 1 then
    for i = 1, #rnn_state_enc do
      rnn_state_enc[i]:zero()
    end
    for t = source_l, 1, -1 do
      local encoder_input = {source_input[t]}
      append_table(encoder_input, rnn_state_enc)
      local out = model[4]:forward(encoder_input)
      rnn_state_enc = out
      context[{{},t}]:add(out[#out])
    end
    if model_opt.init_dec == 1 then
      for L = 1, model_opt.num_layers do
        rnn_state_dec[L*2-1+model_opt.input_feed]:add(rnn_state_enc[L*2-1])
        rnn_state_dec[L*2+model_opt.input_feed]:add(rnn_state_enc[L*2])
      end
    end
  end

  context = context:expand(1, source_l, model_opt.rnn_size)

  -- Start decoder pass
  local context_size = model_opt.rnn_size

  -- Increase context_size if we care about all layers
  if layer == nil or layer < 0 then 
    context_size = context_size * model_opt.num_layers

    -- DEBUG code to extract cell states as well
    -- context_size = context_size * model_opt.num_layers * 2 + context_size
  end

  local encoding = torch.zeros(1, target_l-2, context_size)

  -- local target = target:view(target_l, 1, 1)

  local decoder_input
  for t = 1, target_l do

    local decoder_input
    if model_opt.attn == 1 then
      decoder_input = {target[{{t}}], context[{{1}}], table.unpack(rnn_state_dec)}

      -- Debug statements
      -- print('Target',target[{{t}}])
      -- print('Context',context[{{1}}]:sum())
      -- for i = 1, #init_fwd_dec do
      --   print('In State',i, rnn_state_dec[i]:sum())
      -- end
    else
      decoder_input = {target[t], context[{{}, source_l}], table.unpack(rnn_state_dec)}
    end

    local out = model[2]:forward(decoder_input)

    -- Debug statements
    -- for i = 1, #out do
    --  print('Out:',i,out[i]:sum())
    -- end
    -- print(out)

    -- Save context
    if t ~= 1 and t ~= target_l then
      local final_res
      if layer == nil or layer < 0 then
        -- By default combine all layers
        hidden_states = {}
        for h = 1, #out/2 do
          table.insert(hidden_states, out[h*2])
        end

        -- DEBUG code to extract cell states as well
        --for h = 1, #out do
        --  table.insert(hidden_states, out[h])
        --end
        final_res = torch.cat(hidden_states)
      else
        final_res = out[layer*2]
      end
      encoding[{{},t-1}]:copy(final_res)
    end

    local out_pred_idx = #out
    if model_opt.guided_alignment == 1 then
      out_pred_idx = #out-1
    end
    local next_state = {}
    if model_opt.input_feed == 1 then
      table.insert(next_state, out[out_pred_idx])
    end
    for j = 1, out_pred_idx-1 do
      table.insert(next_state, out[j])
    end
    rnn_state_dec = next_state
  end

  return encoding
end

function getOptions()
  return opt
end

return {
  init = init,
  search = search,
  encode = encode,
  decode = decode,
  search_with_mask = search_with_mask,
  getOptions = getOptions
}
