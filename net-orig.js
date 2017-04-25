
//<![CDATA[

// a few things don't have var in front of them - they update already existing variables the game needs
lanesSide = 0;
patchesAhead = 1;
patchesBehind = 0;
trainIterations = 10000;

var num_inputs = (lanesSide * 2 + 1) * (patchesAhead + patchesBehind);
var num_actions = 5;
var temporal_window = 3;
var network_size = num_inputs * temporal_window + num_actions * temporal_window + num_inputs;

var layer_defs = [];
layer_defs.push({
    type: 'input',
    out_sx: 1,
    out_sy: 1,
    out_depth: network_size
});
layer_defs.push({
    type: 'fc',
    num_neurons: 1,
    activation: 'relu'
});
layer_defs.push({
    type: 'regression',
    num_neurons: num_actions
});

var tdtrainer_options = {
    learning_rate: 0.001,
    momentum: 0.0,
    batch_size: 64,
    l2_decay: 0.01
};

var opt = {};
opt.temporal_window = temporal_window;
opt.experience_size = 3000;
opt.start_learn_threshold = 500;
opt.gamma = 0.7;
opt.learning_steps_total = 10000;
opt.learning_steps_burnin = 1000;
opt.epsilon_min = 0.0;
opt.epsilon_test_time = 0.0;
opt.layer_defs = layer_defs;
opt.tdtrainer_options = tdtrainer_options;

brain = new deepqlearn.Brain(num_inputs, num_actions, opt);

learn = function (state, lastReward) {
    brain.backward(lastReward);
    var action = brain.forward(state);

    draw_net();
    draw_stats();

    return action;
}

//]]>
    
/*###########*/
if (brain) {
brain.value_net.fromJSON({"layers":[{"out_depth":19,"out_sx":1,"out_sy":1,"layer_type":"input"},{"out_depth":1,"out_sx":1,"out_sy":1,"layer_type":"fc","num_inputs":19,"l1_decay_mul":0,"l2_decay_mul":1,"filters":[{"sx":1,"sy":1,"depth":19,"w":{"0":0.11607177193569561,"1":-0.1735173065426127,"2":0.2736317093830827,"3":-0.03387160875448574,"4":0.38749936815292646,"5":-0.1105539003106206,"6":0.0014279856822174753,"7":-0.11275881742927732,"8":0.42413932466816223,"9":0.07183303959768536,"10":0.28923680688978803,"11":0.3136409076597546,"12":-0.10630940243824678,"13":-0.07064942440955971,"14":-0.2664251162511809,"15":0.09285764436815556,"16":-0.17750279804457347,"17":-0.13023287816372503,"18":-0.18003098923791086}}],"biases":{"sx":1,"sy":1,"depth":1,"w":{"0":0.1}}},{"out_depth":1,"out_sx":1,"out_sy":1,"layer_type":"relu"},{"out_depth":5,"out_sx":1,"out_sy":1,"layer_type":"fc","num_inputs":1,"l1_decay_mul":0,"l2_decay_mul":1,"filters":[{"sx":1,"sy":1,"depth":1,"w":{"0":-0.8101064409867743}},{"sx":1,"sy":1,"depth":1,"w":{"0":0.027303835225225684}},{"sx":1,"sy":1,"depth":1,"w":{"0":0.04191038476712376}},{"sx":1,"sy":1,"depth":1,"w":{"0":-0.2755300026260552}},{"sx":1,"sy":1,"depth":1,"w":{"0":-0.09189452655606781}}],"biases":{"sx":1,"sy":1,"depth":5,"w":{"0":0,"1":0,"2":0,"3":0,"4":0}}},{"out_depth":5,"out_sx":1,"out_sy":1,"layer_type":"regression","num_inputs":5}]});
}