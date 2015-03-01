//
//  RBM.cpp
//  OpenGM-RBM
//
//  Created by samuel bean on 2/14/15.
//
//
//This is a test, to make sure I can run code that includes opengm modules
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/inference/gibbs.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdlib.h>

using namespace std;

//*******************
//** Typedefs
//*******************

typedef double
valueType; //Used for factors involving hidden and visible layers

typedef double
HiddenValueType;          // type used for values of hidden layer units

typedef double
VisibleValueType;         //""visible layer units

typedef size_t
IndexType;          // type used for indexing nodes and factors (default : size_t)

typedef size_t
LabelType;          // type used for labels (default : size_t)

typedef opengm::Maximizer
OpType;             // operation used to combine terms

typedef opengm::ExplicitFunction<HiddenValueType,IndexType,LabelType>
HiddenExplicitFunction;   // shortcut for explicite function

typedef opengm::ExplicitFunction<valueType,IndexType,LabelType>
ExplicitFunction;   // shortcut for explicite function

typedef opengm::ExplicitFunction<VisibleValueType,IndexType,LabelType>
VisibleExplicitFunction;   // shortcut for explicite function

typedef opengm::meta::TypeListGenerator<HiddenExplicitFunction>::type
HiddenFunctionTypeList;   // list of all function the model cal use (this trick avoids virtual methods) - here only one

typedef opengm::meta::TypeListGenerator<VisibleExplicitFunction>::type
VisibleFunctionTypeList;   // list of all function the model cal use (this trick avoids virtual methods) - here only one
typedef opengm::DiscreteSpace<IndexType, LabelType>
SpaceType;          // type used to define the feasible statespace

typedef opengm::GraphicalModel< valueType, opengm::Adder, VisibleFunctionTypeList, SpaceType >
Model;              // type of the modelv

typedef opengm::BeliefPropagationUpdateRules< Model, OpType >
UpdateRules;

typedef opengm::MessagePassing< Model, OpType, UpdateRules, opengm::MaxDistance >
BeliefPropagation;

typedef Model::FunctionIdentifier
FunctionIdentifier; // type of the function identifier


int reverseInt (int i)
{
    unsigned char c1, c2, c3, c4;
    
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
void read_mnist_labels( int num_image, int selection, vector< double > &data ){
    ifstream file;
    data.resize( num_image );
    if( selection == 1 )
        file.open("/Applications/of_v0.8.4_osx_release/apps/myApps/RBM/bin/data/train-labels-idx1-ubyte");
    else if( selection == 2 )
        file.open("/Applications/of_v0.8.4_osx_release/apps/myApps/RBM/bin/data/t10k-labels-idx1-ubyte");

    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
        
        
        for(int i=0;i<number_of_images;++i)
        {
            unsigned char temp=0;
            file.read((char*)&temp,sizeof(temp));
            data[ i ] = double( temp );
        }
    }
}


void read_mnist_test_data( int num_image, int data_image, int selection, vector< vector< double > > &data )
{
    //Enter 1 for train data, 2 for train labels, 3 for test data, and 4 for test labels
    ifstream file;
    if( selection == 1 ) file.open("/Applications/of_v0.8.4_osx_release/apps/myApps/RBM/bin/data/train-images-idx3-ubyte");
    else if( selection == 2 ) file.open("/Applications/of_v0.8.4_osx_release/apps/myApps/RBM/bin/data/t10k-images-idx3-ubyte");
    else exit( 0 );
    
    data.resize( num_image, vector< double >( data_image ) );
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
        
        
        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    data[ i ][ ( r * n_rows ) + c ] =  double( temp );
                }
            }
        }
    }
    file.close();
}

int main() {
    //The one layer RBM will have a visible layer that is dependent on the input size
    string train_data_name;
    string train_label_name;
    string test_data_name;
    string test_label_name;
    
    vector< vector< double > > train_data, test_data;
    vector< double > train_labels, test_labels;
    
    
    //Read in the data from the computer, here it is a local path
    //but if this needs to be run on a different computer this can be generalized
    read_mnist_test_data( 60000, 784, 1, train_data );
    read_mnist_labels( 60000, 1, train_labels );
    read_mnist_test_data( 10000, 784, 2, test_data );
    read_mnist_labels( 10000, 2, test_labels );
    
    //Start building the model
    int num_visible_units( 784 ), num_visible_values( 255 ),
                                num_hidden_units( 100 ), num_hidden_values( 2 );
    int num_train_examples( 60000 ), num_test_examples( 10000 );
    vector< double > visible_biases( num_visible_units, 1 / 3 );
    vector< double > hidden_biases( num_hidden_units, 1 / 5 );
    vector< vector< double > > weights;
    weights.resize( num_visible_units, vector< double >( num_hidden_units, 1 / 2 ) );
    
    cout << "Adding variables to graph space" << endl;
    SpaceType space;
    for( int i = 0; i < num_visible_units; ++i )
            space.addVariable( num_visible_values );

    for (int i = 0; i < num_hidden_units; i++)
        space.addVariable( num_hidden_values );
    
    Model gm( space );
    //Models Energy Function
    //E = -sum_i( a_v_i * v_i ) - sum_j( b_h_j * h_j ) - sum_i_j( w_i_j * v_i * h_i )
    
    cout << "Adding visible, unary factors" << endl;
    //Add functions and factors to the graph for visible units and visible bias
    //The unary factors make the sum in term 1 of the energy equation
    for( int i = 0; i < num_visible_units; ++i ) {
        //Initialize the shapes of the unary and binary factors, first term in energy equation
        vector<valueType> shape;
        shape.push_back( num_visible_values );
        ExplicitFunction f( shape.begin(), shape.end() );
        for( LabelType value = 0; value < num_visible_values; ++value ) {
            f( &value ) = train_data[ 0 ][ i ] * visible_biases[ i ];
        }
        FunctionIdentifier id = gm.addFunction( f );
        
        IndexType variableIndex[] = { i };
        gm.addFactor( id, variableIndex, variableIndex + 1 );
        
        shape.clear();
    }
    
    cout << "Adding hidden, unary factors" << endl;
    //Add functions and factors to the graph for hidden units and hidden bias
    //This adds the unary factors, term 2 in the energy function
    for( int i = 0; i < num_hidden_units; ++i ) {
        vector<valueType> shape;
        shape.push_back( num_hidden_values );
        ExplicitFunction f( shape.begin(), shape.end() );
        for( LabelType value = 0; value < num_hidden_values; ++value ) {
            f( &value ) = value * hidden_biases[ i ];
        }
        FunctionIdentifier id = gm.addFunction( f );
        
        IndexType variableIndex[] = { num_visible_units + i };
        gm.addFactor( id, variableIndex, variableIndex + 1 );
        
        shape.clear();
    }
    
    cout << "Adding binary factors" << endl;
    //Need to make binary factors, just a nested for loop of the two blocks of
    //code above
    for( int i = 0; i < num_visible_units; ++i ) {
        for( int j = 0; j < num_hidden_units; ++j ) {
            vector< valueType > binary_shape;
            binary_shape.push_back( num_visible_values ); binary_shape.push_back( num_hidden_values );
            ExplicitFunction bf( binary_shape.begin(), binary_shape.end() );
            LabelType state[] = { 0, 0 };
            
            for(state[0] = 0; state[0] < gm.numberOfLabels( i ); ++state[0]){
                for(state[1] = 0; state[1] < gm.numberOfLabels( num_visible_units + j ); ++state[1]) {
                    bf( state[ 0 ], state[ 1 ] ) = state[ 0 ] * state[ 1 ] * weights.at( i ).at( j );
                    // general function interface
                    //f(state[0], state[1]) = ValueType(rand()) / RAND_MAX; // only works for ExpliciteFunction
                }
            }
            
            IndexType variableIndex[] ={ i, num_visible_units + j };
            FunctionIdentifier bid = gm.addFunction( bf );
            gm.addFactor( bid, variableIndex, variableIndex + 2 );
            
            binary_shape.clear();
        }
    }
    
    cout << "Initializing BP parameters" << endl;
    //Try out belief propogation
    const size_t maxNumberOfIterations = 10;
    const double convergenceBound = 1e-7;
    const double damping = 0;
    cout << "Made it to message passing" << endl;
    BeliefPropagation::Parameter parameter( maxNumberOfIterations, convergenceBound, damping );
    parameter.useNormalization_ = true;
    BeliefPropagation bp( gm, parameter );
    
    BeliefPropagation::VerboseVisitorType visitor;
    
    cout << "Starting inference" << endl;
    bp.infer( visitor );
    
    cout << "inference finished" << endl;
    vector< size_t > labeling( num_visible_units + num_hidden_units );
    bp.arg( labeling );
    
    for( int i = 0; i < labeling.size(); ++i ) cout << "label for " << i << " " << labeling[ i ] << endl;
    
    /*
    //Ability to print is here if needed
    // View some model information
    cout << "The model has " << gm.numberOfVariables() << " variables."<<endl;
    for(size_t i=0; i<gm.numberOfVariables(); ++i){
        cout << " * Variable " << i << " has "<< gm.numberOfLabels(i) << " labels."<<endl;
    }
    cout << "The model has " << gm.numberOfFactors() << " factors."<<endl;
    for(size_t f=0; f<gm.numberOfFactors(); ++f){
        cout << " * Factor " << f << " has order "<< gm[f].numberOfVariables() << "."<<endl;
    }*/
    return 0;
}







