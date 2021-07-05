// import logo from './logo.svg';
// import './App.css';

// function App() {
//   return (
//     <div className="App">
//       <header className="App-header">
//         <img src={logo} className="App-logo" alt="logo" />
//         <p>
//           Edit <code>src/App.js</code> and save to reload.
//         </p>
//         <a
//           className="App-link"
//           href="https://reactjs.org"
//           target="_blank"
//           rel="noopener noreferrer"
//         >
//           Learn React
//         </a>
//       </header>
//     </div>
//   );
// }

// export default App;
// import logo from './logo.svg';
import './App.css';
import React, {useState} from 'react';

function App(){
  // constructor(props) {
  //   super(props);
  //   // this.md = new Remarkable();
  //   this.handleChange = this.handleChange.bind(this);
  //   this.state = { value: "" };
  // }
  const SERVER_URL = 'http://localhost:5000/';
  const [input,setInput] = useState();
  const [result, setResult] = useState([]);
  // function handletextChange(e){
  //   setQuery(e.target.value);
  //   // console.log(query);
  // }

  // function handleSubmit(){
  //   console.log(query)
  // }

  function handleKeyDown(event){
    if (event.key === 'Enter') {
      if(input) {
        setResult([]);
        fetch(`${SERVER_URL}/query?query=`+input)
        .then(res => res.json())
        .then(res_json => {
          setResult(res_json.result);
          console.log(result)
        })
        .catch(err => {
          console.log("why "+err);
        })
      } else {
        setResult([]);
        console.log("no input")
      }
    }
    console.log("hello"+result)
  }

  // getRawMarkup() {
  //   return { __html: this.md.render(this.state.value) };
  // }


  return (
    <div className="Container" >
      <div className="info">
      <p style = {{fontSize:50, color: "white",paddingLeft:'34%'}}><b>Sarcasm Detector</b></p>
      <p style = {{fontSize:25, color: "white",paddingLeft:'70%'}}>Abdul Musawwir 18k-0185
      <br />
      Izaan Sohail &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;18k-0162
      <br />
      Maria Aliasghar &nbsp;&nbsp;18k-0161
      </p>
      </div>
      <div className="info2">
        <p className = "mylabel">
          Enter Your query:
        </p>
        <br/>
        <div>
         <input 
          type="text"
          className="query"
          value={input}
          onKeyDown = {handleKeyDown}
          onChange={e => setInput(e.target.value)}
          style={{height:40,paddingTop:"0%",fontSize:20}}
        />
        </div>
      </div>
      <div className="info2">
      <p style= {{paddingBottom:'0%',color:'white',marginLeft:'40px',fontSize:'x-large'}}>Output:</p>
      <p style = {{color:'white',fontSize:25,marginLeft:'40px',paddingBottom:'12%'}}>{result}</p>
      </div>
      {/* <div
        className="content"
        // dangerouslySetInnerHTML={this.getRawMarkup()}
      />
      {this.state.value} */}
    </div>
  );
}

export default App;
